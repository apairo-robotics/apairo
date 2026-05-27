from __future__ import annotations
from abc import abstractmethod
from pathlib import Path

from apairo.core.abstract_dataset import AbstractDataset
from apairo.core.sample import Sample


class SynchronousDataset(AbstractDataset):
    """Base class for datasets where index ``i`` returns a complete synchronous frame.

    All modalities at index ``i`` are co-captured — no timestamps, no interleaving.
    ``sample.timestamp`` is always ``None``.  Random access and standard PyTorch
    ``DataLoader`` shuffling work without any additional wrappers.

    Subclasses must implement ``__len__``, ``__getitem__``, ``__iter__``, ``__next__``.

    For new synchronous datasets, prefer extending
    :class:`~apairo.core.profiled_dataset.ProfiledDataset` with a YAML profile
    rather than subclassing this directly.

    Attributes:
        timestamps: Always ``None`` — marks this dataset as synchronous.
    """

    timestamps = None

    @property
    def root_dir(self) -> Path:
        return self._root

    def _seq_root(self, path: Path) -> Path:
        """Return the sequence root directory for a native file path.

        Datasets with deeper file structures (e.g. seq/lidar/scan/file.bin)
        should override this to go up the correct number of levels.
        Default: path.parent.parent (one modality directory deep).
        """
        return path.parent.parent

    def derived_path(self, idx: int, key: str, ext: str) -> Path:
        ref = next(iter(self._files.values()))[idx]
        return self._seq_root(ref) / key / f"{ref.stem}.{ext}"

    def _discover_derived(self, key: str, ext: str) -> list[Path]:
        paths = [self.derived_path(i, key, ext) for i in range(len(self))]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Derived key '{key}': {len(missing)} file(s) missing. "
                f"Run run_preprocess(...) to generate them."
            )
        return paths

    def _get_derived_ext(self, seq_dirs: list[Path], key: str) -> str:
        """Look up the ext for a derived key from .apairo — root first, then per-seq dirs."""
        from apairo.core.config import read_config, CONFIG_FILENAME

        for d in [self._root, *seq_dirs]:
            config_path = d / CONFIG_FILENAME
            if config_path.exists():
                config = read_config(d)
                ch = config.get("channels", {}).get(key)
                if ch:
                    loader = ch["loader"]
                    return "npy" if loader in ("npys", "npys_img", "npy") else loader
        raise KeyError(
            f"Key '{key}' not found in .apairo. "
            f"Run run_preprocess(..., output_key='{key}') first."
        )

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Sample: ...

    @abstractmethod
    def __iter__(self): ...

    @abstractmethod
    def __next__(self) -> Sample: ...
