from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml

import numpy as np

from apairo.core.synchronous_dataset import SynchronousDataset
from apairo.core.configurable_dataset import ConfigurableDataset
from apairo.core.config import CONFIG_FILENAME
from apairo.core.sample import Sample
from apairo.loader import DERIVED_LOADERS

_NUMPY_DTYPE: dict[str, type] = {
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "bool": np.bool_,
}

_PROFILES_DIR = Path(__file__).parent.parent / "dataset" / "profiles"

_EXT_TO_LOADER: dict[str, str] = {
    ".bin": "bin",
    ".label": "bin",
    ".npy": "npy",
    ".png": "img",
    ".jpg": "img",
    ".pt": "pt",
}

# Extensions loaded via np.fromfile — raw binary, dtype/reshape/mask are meaningful
_BINARY_EXTS: frozenset[str] = frozenset({".bin", ".label"})


@dataclass
class ModalitySpec:
    ext: str
    dtype: str
    reshape: Optional[list] = None
    mask: Optional[int] = None
    torch_dtype: Optional[str] = None
    loader: Optional[str] = None
    subpath: list[str] = field(default_factory=list)
    optional: bool = False
    resolved_dtype: Optional[type] = field(default=None, compare=False, repr=False)

    @classmethod
    def from_dict(cls, key: str, d: dict) -> "ModalitySpec":
        ext = d["ext"]
        if not ext.startswith("."):
            ext = f".{ext}"
        torch_dtype = d.get("torch_dtype")
        return cls(
            ext=ext,
            dtype=d["dtype"],
            reshape=d.get("reshape"),
            mask=d.get("mask"),
            torch_dtype=torch_dtype,
            loader=d.get("loader"),
            subpath=d.get("subpath", []),
            optional=d.get("optional", False),
            resolved_dtype=_NUMPY_DTYPE.get(torch_dtype) if torch_dtype else None,
        )

    def effective_subpath(self, key: str) -> list[str]:
        return self.subpath if self.subpath else [key]


@dataclass
class LayerSpec:
    type: str
    value: object = None


def _parse_layers(raw: list) -> list[LayerSpec]:
    result = []
    for item in raw:
        if isinstance(item, str):
            result.append(LayerSpec(type=item))
        elif isinstance(item, dict):
            k, v = next(iter(item.items()))
            result.append(LayerSpec(type=k, value=v))
    return result


class ProfiledDataset(SynchronousDataset, ConfigurableDataset):
    """Synchronous dataset driven by a YAML structural profile.

    Subclasses declare a `_profile` class attribute pointing to a YAML file
    (relative to `apairo/dataset/profiles/` or an absolute path).  The profile
    describes the directory layout, file extensions, dtypes, and any type
    transformations.  All file discovery, loading, split filtering, and derived
    key resolution are handled automatically.

    Example:
        Minimal subclass::

            class MyDataset(ProfiledDataset):
                _profile = "my_dataset.yaml"

        Usage::

            ds = MyDataset("/data/my_dataset", keys=["lidar", "labels"], split="train")
            sample = ds[0]
            # sample.data["lidar"]  → np.ndarray
            # sample.data["labels"] → np.ndarray

    Attributes:
        available_keys: Frozenset of key names declared in the profile.
            Populated at class definition time from the YAML file.

    See Also:
        `YAML Profiles <https://apairo.readthedocs.io/datasets/yaml-profiles/>`_
        for the full profile specification.
    """

    _profile: str

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        profile_attr = cls.__dict__.get("_profile")
        if profile_attr:
            p = Path(profile_attr)
            profile_path = p if p.is_absolute() else _PROFILES_DIR / p
            if profile_path.exists():
                with open(profile_path) as f:
                    raw = yaml.safe_load(f)
                cls.available_keys = frozenset(raw.get("modalities", {}).keys())

    def __init__(
        self,
        root_dir: str | Path,
        keys: list[str] | None = None,
        split: str | None = None,
    ) -> None:
        profile_path = (
            Path(self._profile)
            if Path(self._profile).is_absolute()
            else _PROFILES_DIR / self._profile
        )
        with open(profile_path) as f:
            raw = yaml.safe_load(f)

        self._modalities: dict[str, ModalitySpec] = {
            k: ModalitySpec.from_dict(k, v) for k, v in raw["modalities"].items()
        }
        self._layers: list[LayerSpec] = _parse_layers(raw["layers"])

        layer_types = [layer.type for layer in self._layers]
        self._modality_layer_idx: int = layer_types.index("modality")
        seq_idx = (
            layer_types.index("sequence")
            if "sequence" in layer_types
            else len(self._layers) - 1
        )
        self._seq_depth: int = len(self._layers) - seq_idx

        self._root = Path(root_dir)
        self._split_filter = split

        if keys is None:
            keys = [k for k in self._modalities if not self._modalities[k].optional]

        native_keys = [k for k in keys if k in self._modalities]
        derived_keys = [k for k in keys if k not in self._modalities]

        self._set_keys(list(keys))
        self._files: dict[str, list[Path]] = {}

        ref_key: str | None = None
        for key in native_keys:
            files = self._discover_native(key)
            if not files and not self._modalities[key].optional:
                raise FileNotFoundError(
                    f"Key '{key}' declared in profile but no files found under {self._root}."
                )
            self._files[key] = files
            if ref_key is None and files:
                ref_key = key

        lengths = {k: len(v) for k, v in self._files.items() if k in self._modalities}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"Mismatched file counts per key: {lengths}")

        # Detect modality_idx dynamically from first discovered file
        self._modality_idx: int = self._modality_layer_idx
        if ref_key and self._files[ref_key]:
            first = self._files[ref_key][0]
            rel_parts = first.relative_to(self._root).parts
            mapped = self._mapped_name(ref_key)
            if mapped in rel_parts:
                self._modality_idx = rel_parts.index(mapped)

        self._derived_loaders: dict[str, str] = {}
        if derived_keys:
            if ref_key is not None:
                seq_dirs = sorted({self._seq_root(f) for f in self._files[ref_key]})
                for key in derived_keys:
                    ext = self._get_derived_ext(seq_dirs, key)
                    self._derived_loaders[key] = ext
                    self._files[key] = self._discover_derived(key, ext)
            else:
                # No native keys loaded — find seq dirs via .apairo and glob directly.
                # _get_derived_ext raises KeyError if the key is not registered.
                seq_dirs = sorted(p.parent for p in self._root.rglob(CONFIG_FILENAME))
                for key in derived_keys:
                    ext = self._get_derived_ext(seq_dirs, key)
                    self._derived_loaders[key] = ext
                    self._files[key] = self._discover_derived_direct(key, ext)

    def _seq_root(self, path: Path) -> Path:
        d = path
        for _ in range(self._seq_depth):
            d = d.parent
        return d

    def derived_path(self, idx: int, key: str, ext: str) -> Path:
        ref = next(iter(self._files.values()))[idx]
        rel = ref.relative_to(self._root)
        parts = list(rel.parts)
        ref_key = next(iter(self._files))
        src_spec = self._modalities.get(ref_key)
        n = len(src_spec.effective_subpath(ref_key)) if src_spec else 1
        parts[self._modality_idx : self._modality_idx + n] = [key]
        parts[-1] = f"{ref.stem}.{ext}"
        return self._root / Path(*parts)

    def _is_present(self, root_dir: Path, key: str) -> bool:
        spec = self._modalities[key]
        fixed_parts = [layer.value for layer in self._layers if layer.type == "fixed"]
        mapped = self._mapped_name(key)
        if fixed_parts:
            pattern = str(Path(*fixed_parts) / "**" / mapped / f"*{spec.ext}")
        else:
            pattern = f"**/{mapped}/**/*{spec.ext}"
        return any(root_dir.glob(pattern))

    def _bootstrap_config(self, root_dir: Path) -> dict:
        channels = {}
        for key in self.available_keys:
            if self._is_present(root_dir, key):
                spec = self._modalities[key]
                loader = spec.loader or _EXT_TO_LOADER.get(spec.ext, "bin")
                channels[key] = {"loader": loader, "has_timestamps": False}
        return {"version": 1, "channels": channels}

    def _mapped_name(self, key: str) -> str:
        layer = self._layers[self._modality_layer_idx]
        if isinstance(layer.value, dict):
            return layer.value.get(key, key)
        return key

    def _discover_derived_direct(self, key: str, ext: str) -> list[Path]:
        """Glob derived files without a native-key anchor.

        Used when only derived keys are requested and no native files are loaded.
        Applies the split filter the same way as _discover_native.
        """
        files = sorted(self._root.glob(f"**/{key}/**/*.{ext}"))
        if self._split_filter:
            files = [
                f
                for f in files
                if self._split_filter in f.relative_to(self._root).parts
            ]
        if not files:
            raise FileNotFoundError(
                f"Derived key '{key}': no .{ext} files found under '{self._root}'. "
                f"Run run_preprocess(...) to generate them."
            )
        return files

    def _discover_native(self, key: str) -> list[Path]:
        spec = self._modalities[key]
        fixed_parts = [layer.value for layer in self._layers if layer.type == "fixed"]
        mapped = self._mapped_name(key)

        if fixed_parts:
            prefix = Path(*fixed_parts)
            pattern = str(prefix / "**" / mapped / f"*{spec.ext}")
        else:
            pattern = f"**/{mapped}/**/*{spec.ext}"

        files = sorted(self._root.glob(pattern))

        if self._split_filter:
            split_layer = next(
                (layer for layer in self._layers if layer.type == "split"), None
            )
            if split_layer is not None:
                files = [
                    f
                    for f in files
                    if self._split_filter in f.relative_to(self._root).parts
                ]
        return files

    def __len__(self) -> int:
        if not self._files:
            return 0
        return len(next(iter(self._files.values())))

    def __getitem__(self, idx: int) -> Sample:
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        data: dict = {}
        for key in self._keys:
            path = self._files[key][idx]
            if key in self._modalities:
                spec = self._modalities[key]
                if spec.ext in _BINARY_EXTS:
                    arr = np.fromfile(path, dtype=np.dtype(spec.dtype))
                    if spec.reshape:
                        arr = arr.reshape(spec.reshape)
                    if spec.mask is not None:
                        arr &= spec.mask
                else:
                    loader_name = spec.loader or _EXT_TO_LOADER.get(spec.ext)
                    if loader_name is None or loader_name not in DERIVED_LOADERS:
                        raise ValueError(
                            f"No loader for extension '{spec.ext}' on key '{key}'. "
                            f"Set 'loader' in the profile or use a supported extension."
                        )
                    arr = DERIVED_LOADERS[loader_name](path)
                    if spec.reshape:
                        arr = arr.reshape(spec.reshape)
                    if spec.mask is not None:
                        arr &= spec.mask
                if spec.resolved_dtype is not None:
                    arr = arr.astype(spec.resolved_dtype)
                data[key] = arr
            else:
                ext = self._derived_loaders[key]
                data[key] = DERIVED_LOADERS[ext](path)
        return Sample(data=data)

    def __iter__(self):
        self._iter_pos = 0
        return self

    def __next__(self) -> Sample:
        if self._iter_pos >= len(self):
            raise StopIteration
        sample = self[self._iter_pos]
        self._iter_pos += 1
        return sample
