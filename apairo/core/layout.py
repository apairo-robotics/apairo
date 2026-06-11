"""Dataset layout — single source of truth for on-disk organization.

A :class:`DatasetLayout` describes how a dataset *family* encodes its
channels on disk: where each channel lives, in which store format, with
which dtype, chunking and naming policies.  Both the read path (datasets
building loaders) and the write path (pipelines persisting missions)
consume the same object, so they cannot drift apart.

Boundaries:

* the layout describes **encoding** (where / how), never *what exists* in a
  given mission — that is per-instance state (``.apairo``, directory scan);
* loaders and writers stay pure format mechanics — the layout *configures*
  them with the dataset's policies (naming, chunks, compression);
* unknown channels resolve through the **flat convention**
  (``<root>/<key>.zarr``) with the layout's default spec, so the channel
  table never grows for new channels.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ChannelSpec:
    """Encoding of one channel.

    Args:
        path: Sub-path components relative to the dataset root.
        store: ``"zarr"`` (array store) or ``"tar_jpeg"`` (image archive).
        dtype: Dtype imposed at write time (``None`` keeps the input dtype).
        chunks: Chunking policy, called with the array shape at write time.
        name_to_index: Member name -> frame index policy (``tar_jpeg``).
        member_name: Frame index -> member name policy (``tar_jpeg``).
        write_options: Extra writer arguments (e.g. ``{"quality": 95}``).
    """

    path: Tuple[str, ...]
    store: str = "zarr"
    dtype: Optional[np.dtype] = None
    chunks: Optional[Callable[[tuple], tuple]] = None
    name_to_index: Optional[Callable[[str], Optional[int]]] = None
    member_name: Optional[Callable[[int], str]] = None
    write_options: dict = field(default_factory=dict)


class DatasetLayout:
    """On-disk organization of a dataset family.

    Args:
        channels: Channel key -> :class:`ChannelSpec` for the channels whose
            location is fixed by the format (the legacy/grouped part).
        compression: Optional ``(codec, level)`` Blosc compression applied to
            every zarr channel (e.g. ``("zstd", 5)``).
        default: Spec template for channels resolved by the flat convention
            (its ``path`` is ignored and replaced by ``<key>.zarr``).
    """

    def __init__(
        self,
        channels: Dict[str, ChannelSpec],
        compression: Optional[Tuple[str, int]] = None,
        default: Optional[ChannelSpec] = None,
    ) -> None:
        self._channels = dict(channels)
        self._compression = compression
        self._default = default or ChannelSpec(path=())

    # ------------------------------------------------------------ resolution

    @property
    def channels(self) -> Dict[str, ChannelSpec]:
        return dict(self._channels)

    def spec(self, key: str) -> ChannelSpec:
        """Spec for *key*: table entry, or flat-convention default."""
        if key in self._channels:
            return self._channels[key]
        return replace(self._default, path=(f"{key}.zarr",))

    def path(self, root: Path | str, key: str) -> Path:
        """Absolute path of channel *key* under *root*."""
        return Path(root) / Path(*self.spec(key).path)

    def exists(self, root: Path | str, key: str) -> bool:
        spec = self.spec(key)
        p = self.path(root, key)
        return p.is_file() if spec.store == "tar_jpeg" else p.is_dir()

    def scan(self, root: Path | str) -> List[str]:
        """Channel keys present in *root*: table channels + flat convention."""
        root = Path(root)
        keys = [k for k in sorted(self._channels) if self.exists(root, k)]
        table_roots = {spec.path[0] for spec in self._channels.values() if spec.path}
        keys += sorted(
            child.name[: -len(".zarr")]
            for child in root.glob("*.zarr")
            if child.is_dir() and child.name not in table_roots
        )
        return keys

    # ------------------------------------------------------------- read path

    def loader(self, root: Path | str, key: str, n_frames: int):
        """Configured loader for *key*, or ``None`` when absent on disk."""
        spec = self.spec(key)
        p = self.path(root, key)

        if spec.store == "tar_jpeg":
            if not p.is_file():
                return None
            from apairo.loader.tar_loader import TarImageLoader
            return TarImageLoader(p, n_frames, spec.name_to_index)

        if not p.is_dir():
            return None
        from apairo.loader.zarr_loader import ZarrLoader
        return ZarrLoader(p)

    # ------------------------------------------------------------ write path

    def write(self, root: Path | str, key: str, data: np.ndarray) -> Path:
        """Persist one channel under its spec'd path and return that path."""
        spec = self.spec(key)
        p = self.path(root, key)

        if spec.store == "tar_jpeg":
            from apairo.writer import TarImageWriter
            TarImageWriter(
                member_name=spec.member_name, **spec.write_options
            ).write(data, p)
            return p

        from apairo.writer import ZarrWriter
        data = np.asarray(data, dtype=spec.dtype) if spec.dtype else np.asarray(data)
        codec, level = self._compression or (None, 5)
        ZarrWriter(
            chunks=spec.chunks(data.shape) if spec.chunks else None,
            compression=codec,
            compression_level=level,
            **spec.write_options,
        ).write(data, p)
        return p
