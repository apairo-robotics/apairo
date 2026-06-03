from __future__ import annotations
from pathlib import Path
import numpy as np


class ZarrWriter:
    """Write a numpy array to a Zarr array store.

    The full array is written in one call (sequence-level).  For per-frame
    incremental writes, accumulate frames first then call :meth:`write` once.
    """

    def write(self, data: np.ndarray, path: Path) -> None:
        """Save *data* as a Zarr array at *path*.

        Args:
            data: Array to persist (any shape / dtype).
            path: Destination Zarr store path (e.g. ``points.zarr``).
        """
        try:
            import zarr
        except ImportError:
            raise ImportError(
                "zarr is required for ZarrWriter. "
                "Install with: pip install zarr"
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        zarr.save_array(str(path), data)
