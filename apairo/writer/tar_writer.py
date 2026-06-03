from __future__ import annotations
import tarfile
from io import BytesIO
from pathlib import Path
import numpy as np


class TarImageWriter:
    """Write image frames to a tar archive.

    Accepts a single image ``(H, W[, C])`` or a batch ``(N, H, W[, C])``.
    Frames are stored as JPEG members named ``0.jpg``, ``1.jpg``, … matching
    the indexing convention expected by
    :class:`~apairo.loader.tar_loader.TarImageLoader`.

    The full archive is written in one call (sequence-level).
    """

    def write(self, data: np.ndarray, path: Path) -> None:
        """Save *data* as a tar archive of JPEG images at *path*.

        Args:
            data: ``uint8`` array of shape ``(H, W, C)``, ``(H, W)``, or
                ``(N, H, W, C)`` / ``(N, H, W)``.
            path: Destination tar file path (e.g. ``images.tar``).
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError(
                "Pillow is required for TarImageWriter. "
                "Install with: pip install Pillow"
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        frames = data[None] if data.ndim in (2, 3) else data  # → (N, H, W[, C])

        with tarfile.open(str(path), "w") as tf:
            for i, frame in enumerate(frames):
                img = Image.fromarray(frame.astype(np.uint8))
                buf = BytesIO()
                img.save(buf, format="JPEG")
                raw = buf.getvalue()
                info = tarfile.TarInfo(name=f"{i}.jpg")
                info.size = len(raw)
                tf.addfile(info, BytesIO(raw))
