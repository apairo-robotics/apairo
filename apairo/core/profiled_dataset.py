from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional
import yaml

if TYPE_CHECKING:
    from apairo.core.sequence_view import SequenceView

import numpy as np

from apairo.core.synchronous_dataset import SynchronousDataset
from apairo.core.configurable_dataset import ConfigurableDataset
from apairo.core.sample import Sample
from apairo.loader import DERIVED_LOADERS, TXTLoader

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
}

_BINARY_EXTS: frozenset[str] = frozenset({".bin", ".label"})

# Loader names that map to a single sequence-level file (one file, N rows).
_SEQUENCE_LOADERS: frozenset[str] = frozenset({"txt_rows"})


@dataclass
class SplitSpec:
    type: str
    files: dict[str, str]  # split_name -> relative path to the lst file


def _parse_splits_spec(raw: dict) -> "SplitSpec | None":
    if not raw:
        return None
    split_type = raw.get("type")
    if not split_type:
        return None
    return SplitSpec(
        type=split_type, files={k: v for k, v in raw.items() if k != "type"}
    )


def _read_lst_frame_set(lst_path: Path) -> set[tuple[str, str]]:
    """Parse a .lst split file into a set of (seq_id, stem) pairs.

    Each non-empty line is expected to have space-separated columns; the first
    column is a relative path of the form ``<seq>/<modality_dir>/<stem>.ext``.
    """
    result: set[tuple[str, str]] = set()
    with open(lst_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            first_col = line.split()[0]
            parts = Path(first_col).parts
            seq_id = parts[0]
            stem = Path(parts[-1]).stem
            result.add((seq_id, stem))
    return result


@dataclass
class ModalitySpec:
    ext: str
    dtype: Optional[str] = None
    reshape: Optional[list] = None
    mask: Optional[int] = None
    torch_dtype: Optional[str] = None
    loader: Optional[str] = None
    subpath: list[str] = field(default_factory=list)
    optional: bool = False
    resolved_dtype: Optional[type] = field(default=None, compare=False, repr=False)

    @classmethod
    def from_dict(cls, key: str, d: dict) -> "ModalitySpec":
        ext = d.get("ext", "")
        if ext and not ext.startswith("."):
            ext = f".{ext}"
        torch_dtype = d.get("torch_dtype")
        return cls(
            ext=ext,
            dtype=d.get("dtype"),
            reshape=d.get("reshape"),
            mask=d.get("mask"),
            torch_dtype=torch_dtype,
            loader=d.get("loader"),
            subpath=d.get("subpath", []),
            optional=d.get("optional", False),
            resolved_dtype=_NUMPY_DTYPE.get(torch_dtype) if torch_dtype else None,
        )

    @property
    def is_sequence_file(self) -> bool:
        return self.loader in _SEQUENCE_LOADERS

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


class _PerFrameLoader:
    """Wraps a sorted list of per-frame file paths and handles loading."""

    def __init__(self, paths: list[Path], spec: ModalitySpec) -> None:
        self.paths = paths
        self._spec = spec

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> np.ndarray:
        path = self.paths[idx]
        spec = self._spec
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
                    f"No loader for extension '{spec.ext}'. "
                    f"Set 'loader' in the profile or use a supported extension."
                )
            arr = DERIVED_LOADERS[loader_name](path)
            if spec.reshape:
                arr = arr.reshape(spec.reshape)
            if spec.mask is not None:
                arr &= spec.mask
        if spec.resolved_dtype is not None:
            arr = arr.astype(spec.resolved_dtype)
        return arr


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
            # sample.data["lidar"]  -> np.ndarray
            # sample.data["labels"] -> np.ndarray

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
        sequence_ids: list[str] | None = None,
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
        self._sequence_ids_filter: frozenset[str] | None = (
            frozenset(sequence_ids) if sequence_ids is not None else None
        )

        self._splits_spec: SplitSpec | None = _parse_splits_spec(raw.get("splits", {}))
        self._frame_filter: set[tuple[str, str]] | None = None
        if (
            split is not None
            and self._splits_spec is not None
            and self._splits_spec.type == "lst"
        ):
            lst_rel = self._splits_spec.files.get(split)
            if lst_rel is None:
                available = list(self._splits_spec.files.keys())
                raise ValueError(
                    f"Split '{split}' not declared in profile. Available: {available}"
                )
            self._frame_filter = _read_lst_frame_set(self._root / lst_rel)

        # .apairo is the source of truth: raw channels present + preprocessed channels created.
        config = self._load_or_create_config(self._root)
        channels: dict = config.get("channels", {})

        if keys is None:
            keys = [
                k
                for k, v in channels.items()
                if v.get("kind", "raw") == "raw" and not self._modalities[k].optional
            ]

        # Classify each requested key.
        raw_keys: list[str] = []
        derived_keys: list[str] = []
        for k in keys:
            ch = channels.get(k)
            if ch is None:
                # Not in .apairo — allow if it is a profile key (raw, not yet scanned).
                if k not in self._modalities:
                    raise KeyError(
                        f"Key '{k}' is not available in '{self._root}'. "
                        f"Available: {sorted(channels)}. "
                        f"Register preprocessed channels with "
                        f"{type(self).__name__}.register_channel()."
                    )
                raw_keys.append(k)
            elif ch.get("kind", "raw") == "raw":
                raw_keys.append(k)
            else:
                derived_keys.append(k)

        self._set_keys(list(keys))
        self._files: dict[str, list[Path]] = {}
        self._loaders: dict[str, _PerFrameLoader | TXTLoader] = {}
        self._ref_key: str | None = None

        for key in raw_keys:
            spec = self._modalities[key]
            if spec.is_sequence_file:
                paths = self._discover_sequence_files(key)
                if not paths and not spec.optional:
                    raise FileNotFoundError(
                        f"Key '{key}': no '{self._mapped_name(key)}{spec.ext}' "
                        f"files found under {self._root}."
                    )
                if paths:
                    self._loaders[key] = TXTLoader(paths, spec.reshape)
            else:
                paths = self._discover_native(key)
                if not paths and not spec.optional:
                    raise FileNotFoundError(
                        f"Key '{key}' declared in profile but no files found under {self._root}."
                    )
                if paths:
                    self._files[key] = paths
                    self._loaders[key] = _PerFrameLoader(paths, spec)
                    if self._ref_key is None:
                        self._ref_key = key

        frame_counts = {k: len(v) for k, v in self._loaders.items()}
        if len(set(frame_counts.values())) > 1:
            raise ValueError(f"Mismatched frame counts per key: {frame_counts}")

        self._modality_idx: int = self._modality_layer_idx
        if self._ref_key and self._files.get(self._ref_key):
            first = self._files[self._ref_key][0]
            rel_parts = first.relative_to(self._root).parts
            mapped = self._mapped_name(self._ref_key)
            if mapped in rel_parts:
                self._modality_idx = rel_parts.index(mapped)

        for key in derived_keys:
            loader = channels[key]["loader"]
            ext = "npy" if loader in ("npys", "npys_img", "npy") else loader
            paths = self._discover_derived(key, ext)
            spec = ModalitySpec(ext=f".{ext}", loader=ext)
            self._loaders[key] = _PerFrameLoader(paths, spec)

        # If no native key was loaded (e.g. preprocessing a derived channel),
        # fall back to the first derived key as the path reference so that
        # derived_path() can resolve output locations.
        if self._ref_key is None:
            for key in derived_keys:
                loader = self._loaders.get(key)
                if loader is not None and loader.paths:
                    self._files[key] = loader.paths
                    self._ref_key = key
                    first = self._files[self._ref_key][0]
                    rel_parts = first.relative_to(self._root).parts
                    mapped = self._mapped_name(self._ref_key)
                    if mapped in rel_parts:
                        self._modality_idx = rel_parts.index(mapped)
                    break

        self._set_keys([k for k in keys if k in self._loaders])

        self._seq_groups: dict[str, list[int]] = {}
        anchor = (
            self._files.get(self._ref_key)
            if self._ref_key
            else next(
                (
                    v.paths
                    for v in self._loaders.values()
                    if isinstance(v, _PerFrameLoader)
                ),
                None,
            )
        )
        if anchor:
            for i, path in enumerate(anchor):
                seq_name = self._seq_root(path).name
                self._seq_groups.setdefault(seq_name, []).append(i)

    def _seq_root(self, path: Path) -> Path:
        d = path
        for _ in range(self._seq_depth):
            d = d.parent
        return d

    def derived_path(self, idx: int, key: str, ext: str) -> Path:
        ref = self._files[self._ref_key][idx]
        rel = ref.relative_to(self._root)
        parts = list(rel.parts)
        src_spec = self._modalities.get(self._ref_key)
        n = len(src_spec.effective_subpath(self._ref_key)) if src_spec else 1
        parts[self._modality_idx : self._modality_idx + n] = [key]
        parts[-1] = f"{ref.stem}.{ext}"
        return self._root / Path(*parts)

    def _is_present(self, root_dir: Path, key: str) -> bool:
        spec = self._modalities[key]
        mapped = self._mapped_name(key)
        fixed_parts = [layer.value for layer in self._layers if layer.type == "fixed"]
        if spec.is_sequence_file:
            return any(root_dir.glob(f"**/{mapped}{spec.ext}"))
        if fixed_parts:
            prefix = Path(*fixed_parts)
            return any(root_dir.glob(str(prefix / "**" / mapped / f"*{spec.ext}")))
        return any(root_dir.glob(f"**/{mapped}/**/*{spec.ext}"))

    def _bootstrap_config(self, root_dir: Path) -> dict:
        channels = {}
        for key in sorted(self.available_keys):
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

    def _discover_sequence_files(self, key: str) -> list[Path]:
        """Find sequence-level files (one per sequence, not per frame)."""
        spec = self._modalities[key]
        fixed_parts = [layer.value for layer in self._layers if layer.type == "fixed"]
        mapped = self._mapped_name(key)

        if fixed_parts:
            prefix = Path(*fixed_parts)
            pattern = str(prefix / f"**/{mapped}{spec.ext}")
        else:
            pattern = f"**/{mapped}{spec.ext}"

        paths = sorted(self._root.glob(pattern))
        if self._sequence_ids_filter is not None:
            paths = [p for p in paths if p.parent.name in self._sequence_ids_filter]
        return paths

    def _discover_derived(self, key: str, ext: str) -> list[Path]:
        fixed_parts = [layer.value for layer in self._layers if layer.type == "fixed"]
        if fixed_parts:
            prefix = Path(*fixed_parts)
            pattern = str(prefix / "**" / key / f"*.{ext}")
        else:
            pattern = f"**/{key}/**/*.{ext}"

        files = sorted(self._root.glob(pattern))
        if self._split_filter:
            files = [
                f
                for f in files
                if self._split_filter in f.relative_to(self._root).parts
            ]
        if self._sequence_ids_filter is not None:
            files = [
                f for f in files if self._seq_root(f).name in self._sequence_ids_filter
            ]
        if self._frame_filter is not None:
            files = [
                f
                for f in files
                if (self._seq_root(f).name, f.stem) in self._frame_filter
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
        if self._sequence_ids_filter is not None:
            files = [
                f for f in files if self._seq_root(f).name in self._sequence_ids_filter
            ]
        if self._frame_filter is not None:
            files = [
                f
                for f in files
                if (self._seq_root(f).name, f.stem) in self._frame_filter
            ]
        return files

    def __len__(self) -> int:
        if not self._loaders:
            return 0
        return len(next(iter(self._loaders.values())))

    def describe(self, sequence_id: str | None = None) -> dict:
        """Describe available channels for this dataset.

        Reads ``.apairo`` at the dataset root (creating it if absent) and
        cross-references it with the profile's declared modalities to show
        which raw channels are present or missing, and which preprocessed
        channels have been registered.

        Args:
            sequence_id: Optional sequence identifier -- used as the display
                label only. Channel availability is dataset-wide.

        Returns:
            ``{"raw": {"present": [...], "missing": [...]}, "preprocess": {...}}``

        Example::

            ds = Rellis3DDataset("/data/RELLIS")
            ds.describe("00000")
        """
        from apairo.core.config import config_exists, read_config

        # Raw channels: probe filesystem directly — .apairo only stores preprocessed ones.
        raw_present = sorted(
            k for k in self.available_keys if self._is_present(self._root, k)
        )
        raw_missing = sorted(
            k for k in self.available_keys if not self._is_present(self._root, k)
        )

        preprocess = {}
        if config_exists(self._root):
            config = read_config(self._root)
            preprocess = {
                k: v
                for k, v in config.get("channels", {}).items()
                if v.get("kind") == "preprocess"
            }

        label = sequence_id if sequence_id is not None else self._root.name
        print(f"\n{type(self).__name__} -- {label}")
        print("─" * 50)
        print("Raw channels")
        if raw_present:
            print("  present  :", ", ".join(raw_present))
        if raw_missing:
            print("  missing  :", ", ".join(raw_missing))
        if not raw_present and not raw_missing:
            print("  (none)")
        print("Preprocessed channels")
        if preprocess:
            for key, meta in sorted(preprocess.items()):
                ts_info = (
                    f"<- timestamps from {meta['timestamps_from']}"
                    if "timestamps_from" in meta
                    else "<- own timestamps"
                )
                src_info = (
                    f"  sources: {meta['sources']}" if meta.get("sources") else ""
                )
                print(f"  {key:<20} {meta['loader']:<6} {ts_info}{src_info}")
        else:
            print("  (none)")
        print()
        return {
            "raw": {"present": raw_present, "missing": raw_missing},
            "preprocess": preprocess,
        }

    @property
    def splits(self) -> list[str]:
        if self._splits_spec is not None:
            return list(self._splits_spec.files.keys())
        for layer in self._layers:
            if layer.type == "split" and isinstance(layer.value, list):
                return list(layer.value)
        return []

    def split(self, name: str) -> "ProfiledDataset":
        """Return a new dataset instance filtered to the named split."""
        return type(self)(
            self._root,
            keys=list(self._keys),
            split=name,
            sequence_ids=list(self._sequence_ids_filter)
            if self._sequence_ids_filter
            else None,
        )

    @property
    def sequence_ids(self) -> list[str]:
        return list(self._seq_groups.keys())

    @property
    def frame_sequence_ids(self) -> np.ndarray:
        """Sequence ID for every frame, indexed by global frame index.

        Returns a string array of shape ``(len(self),)`` where
        ``frame_sequence_ids[i]`` is the sequence ID that frame ``i`` belongs
        to.  Combined with :attr:`FilteredView.indices`, this lets you split a
        pre-filtered dataset by sequence without a second disk sweep::

            ds_filtered = ds.filter("trav_gt", HasMinPositives(min_pos))
            seq_ids = ds.frame_sequence_ids[ds_filtered.indices]

            for train_seqs, val_seqs in folds:
                train_idx = np.where(np.isin(seq_ids, train_seqs))[0]
                val_idx   = np.where(np.isin(seq_ids, val_seqs))[0]
                ds_train  = ds_filtered.filter(train_idx)
                ds_val    = ds_filtered.filter(val_idx)
        """
        result = np.empty(len(self), dtype=object)
        for seq_id, indices in self._seq_groups.items():
            result[indices] = seq_id
        return result

    def sequences(self) -> "list[SequenceView]":
        from apairo.core.sequence_view import SequenceView  # noqa: F401

        return [self.sequence(sid) for sid in self.sequence_ids]

    def sequence(self, seq_id: str) -> "SequenceView":
        if seq_id not in self._seq_groups:
            raise KeyError(
                f"Sequence '{seq_id}' not found. " f"Available: {self.sequence_ids}"
            )
        from apairo.core.sequence_view import SequenceView

        return SequenceView(self, self._seq_groups[seq_id], seq_id)

    def _load(self, idx) -> Sample:
        if isinstance(idx, tuple):
            seq_id, local_idx = idx
            view = self.sequence(seq_id)
            return self._load(view._indices[local_idx])
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        return Sample(data={key: self._loaders[key][idx] for key in self._keys})

