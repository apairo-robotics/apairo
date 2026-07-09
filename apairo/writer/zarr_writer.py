from __future__ import annotations

from pathlib import Path

import numpy as np


class ZarrWriter:
    """Write a numpy array to a Zarr array store.

    The full array is written in one call (sequence-level).  For per-frame
    incremental writes, accumulate frames first then call :meth:`write` once.

    By default the array is saved with Zarr's standard settings.  Datasets
    that care about on-disk layout can impose chunking and Blosc compression:

    Args:
        chunks: Chunk shape (dataset-imposed; Zarr default when omitted).
        compression: Blosc codec name (e.g. ``"zstd"``, ``"lz4"``).
            ``None`` (default) keeps Zarr's standard codec.
        compression_level: Blosc compression level (used with *compression*).
    """

    def __init__(
        self,
        chunks: tuple[int, ...] | None = None,
        compression: str | None = None,
        compression_level: int = 5,
    ) -> None:
        self._chunks = chunks
        self._compression = compression
        self._compression_level = compression_level

    def write(self, data: np.ndarray, path: Path) -> None:
        """Save *data* as a Zarr array at *path*.

        Args:
            data: Array to persist (any shape / dtype).
            path: Destination Zarr store path (e.g. ``points.zarr``).
        """
        try:
            import zarr
        except ImportError as exc:
            raise ImportError(
                "zarr is required for ZarrWriter. Install with: pip install zarr"
            ) from exc
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self._chunks is None and self._compression is None:
            zarr.save_array(str(path), data)
            return

        compressor = None
        if self._compression is not None:
            from numcodecs import Blosc

            compressor = Blosc(
                cname=self._compression,
                clevel=self._compression_level,
                shuffle=Blosc.SHUFFLE,
            )

        arr = zarr.create(
            store=zarr.storage.LocalStore(str(path)),
            shape=data.shape,
            chunks=self._chunks or True,
            dtype=data.dtype,
            compressor=compressor,
            overwrite=True,
            zarr_format=2,
        )
        arr[:] = data
