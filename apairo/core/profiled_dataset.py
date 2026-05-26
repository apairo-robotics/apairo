from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml

import numpy as np
import torch

from apairo.core.synchronous_dataset import SynchronousDataset
from apairo.core.configurable_dataset import ConfigurableDataset
from apairo.core.sample import Sample
from apairo.loader import DERIVED_LOADERS

_PROFILES_DIR = Path(__file__).parent.parent / "dataset" / "profiles"

_EXT_TO_LOADER: dict[str, str] = {
    ".bin":   "bin",
    ".label": "bin",
    ".npy":   "npy",
    ".png":   "img",
    ".jpg":   "img",
    ".pt":    "pt",
}


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

    @classmethod
    def from_dict(cls, key: str, d: dict) -> "ModalitySpec":
        ext = d["ext"]
        if not ext.startswith("."):
            ext = f".{ext}"
        return cls(
            ext=ext,
            dtype=d["dtype"],
            reshape=d.get("reshape"),
            mask=d.get("mask"),
            torch_dtype=d.get("torch_dtype"),
            loader=d.get("loader"),
            subpath=d.get("subpath", []),
            optional=d.get("optional", False),
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
    """Dataset driven by a YAML structural profile.

    Subclasses set _profile = "filename.yaml" (relative to profiles dir)
    or an absolute path.
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
            k: ModalitySpec.from_dict(k, v)
            for k, v in raw["modalities"].items()
        }
        self._layers: list[LayerSpec] = _parse_layers(raw["layers"])

        layer_types = [l.type for l in self._layers]
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

        if derived_keys and not native_keys:
            raise KeyError(
                f"Derived keys {derived_keys} require at least one native key "
                f"({sorted(self._modalities)}) alongside them."
            )

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
        # Derived key support added in Task 6

    def _mapped_name(self, key: str) -> str:
        layer = self._layers[self._modality_layer_idx]
        if isinstance(layer.value, dict):
            return layer.value.get(key, key)
        return key

    def _discover_native(self, key: str) -> list[Path]:
        spec = self._modalities[key]
        fixed_parts = [l.value for l in self._layers if l.type == "fixed"]
        mapped = self._mapped_name(key)

        if fixed_parts:
            prefix = Path(*fixed_parts)
            pattern = str(prefix / "**" / mapped / f"*{spec.ext}")
        else:
            pattern = f"**/{mapped}/**/*{spec.ext}"

        files = sorted(self._root.glob(pattern))

        if self._split_filter and any(l.type == "split" for l in self._layers):
            split_idx = next(i for i, l in enumerate(self._layers) if l.type == "split")
            files = [
                f for f in files
                if split_idx < len(f.relative_to(self._root).parts)
                and f.relative_to(self._root).parts[split_idx] == self._split_filter
            ]
        return files

    def __len__(self) -> int:
        return len(next(iter(self._files.values())))
