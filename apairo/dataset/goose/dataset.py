from __future__ import annotations
from pathlib import Path

import numpy as np
import torch

from apairo.core.synchronous_dataset import SynchronousDataset
from apairo.core.configurable_dataset import ConfigurableDataset
from apairo.core.sample import Sample
from apairo.loader import DERIVED_LOADERS

_AVAILABLE_KEYS = {"lidar", "labels"}


class Goose3DDataset(SynchronousDataset, ConfigurableDataset):
    available_keys = frozenset(_AVAILABLE_KEYS)
    r"""Goose3D synchronous dataset (LiDAR + semantic labels).

    Args:
        root_dir: Dataset root. LiDAR files are discovered via
            ``**/lidar/**/*.bin`` and labels via ``**/labels/**/*.label``.
        keys: Modalities to load — any subset of ``{"lidar", "labels"}``.
    """

    def __init__(self, root_dir: str | Path, keys: list[str] | None = None) -> None:
        if keys is None:
            keys = ["lidar", "labels"]
        root = Path(root_dir)
        self._root = root

        native_keys = [k for k in keys if k in _AVAILABLE_KEYS]
        derived_keys = [k for k in keys if k not in _AVAILABLE_KEYS]

        if derived_keys and not native_keys:
            raise KeyError(
                f"Derived keys {derived_keys} require at least one native key "
                f"({sorted(_AVAILABLE_KEYS)}) alongside them."
            )

        self._set_keys(list(keys))
        self._files: dict[str, list[Path]] = {}
        if "lidar" in self._keys:
            self._files["lidar"] = sorted(root.glob("**/lidar/**/*.bin"))
        if "labels" in self._keys:
            self._files["labels"] = sorted(root.glob("**/labels/**/*.label"))
        lengths = {k: len(v) for k, v in self._files.items() if k in _AVAILABLE_KEYS}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"Mismatched file counts per key: {lengths}")

        # Determine where the modality dir sits in the path (varies with root depth).
        # e.g. root=GOOSE_3D/       → train/lidar/train/seq/file → idx=1
        #      root=GOOSE_3D/train/ → lidar/train/seq/file       → idx=0
        ref_key = native_keys[0] if native_keys else None
        if ref_key and self._files.get(ref_key):
            first = self._files[ref_key][0]
            rel_parts = first.relative_to(root).parts
            self._modality_idx: int = rel_parts.index(ref_key)
        else:
            self._modality_idx = 0

        self._derived_loaders: dict[str, str] = {}
        if derived_keys:
            ref_files = self._files[native_keys[0]]
            seq_dirs = sorted({self._seq_root(f) for f in ref_files})
            for key in derived_keys:
                ext = self._get_derived_ext(seq_dirs, key)
                self._derived_loaders[key] = ext
                self._files[key] = self._discover_derived(key, ext)

    def __len__(self) -> int:
        return len(next(iter(self._files.values())))

    def __getitem__(self, idx: int) -> Sample:
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        data: dict = {}
        for key in self._keys:
            path = self._files[key][idx]
            if key == "lidar":
                arr = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
                data[key] = torch.from_numpy(arr)
            elif key == "labels":
                arr = np.fromfile(path, dtype=np.int32)
                data[key] = torch.from_numpy(arr).long()
            else:
                ext = self._derived_loaders[key]
                data[key] = DERIVED_LOADERS[ext](path)
        return Sample(data=data)

    def derived_path(self, idx: int, key: str, ext: str) -> Path:
        # Replace the modality component with the derived key name.
        # Works regardless of root depth (GOOSE_3D/ or GOOSE_3D/train/).
        ref = next(iter(self._files.values()))[idx]
        rel = ref.relative_to(self._root)
        parts = list(rel.parts)
        parts[self._modality_idx] = key
        parts[-1] = f"{ref.stem}.{ext}"
        return self._root / Path(*parts)

    def _bootstrap_config(self, sequence_dir: Path) -> dict:
        channels = {
            key: {"loader": "bin", "has_timestamps": False}
            for key in sorted(_AVAILABLE_KEYS)
            if (sequence_dir / key).is_dir()
        }
        return {"version": 1, "channels": channels}

    def __iter__(self):
        self._iter_pos = 0
        return self

    def __next__(self) -> Sample:
        if self._iter_pos >= len(self):
            raise StopIteration
        sample = self[self._iter_pos]
        self._iter_pos += 1
        return sample
