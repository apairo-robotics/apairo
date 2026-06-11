from __future__ import annotations
import tarfile
from io import BytesIO
from pathlib import Path
from typing import Callable, Optional
import numpy as np


class TarImageWriter:
    """Write image frames to a tar archive.

    Accepts a single image ``(H, W[, C])`` or a batch ``(N, H, W[, C])``.
    The writer only handles tar mechanics; member naming is a dataset
    concern, imposed via ``member_name`` (frame index -> member name).
    When the writer is instantiated generically (e.g. through the
    ``WRITERS`` registry), the integer policy ``<i>.jpg`` is used.

    The full archive is written in one call (sequence-level).
    """

    def __init__(
        self,
        member_name: Optional[Callable[[int], str]] = None,
        quality: Optional[int] = None,
    ) -> None:
        self._member_name = member_name or (lambda i: f"{i}.jpg")
        self._quality = quality

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
                if self._quality is not None:
                    img.save(buf, format="JPEG", quality=self._quality)
                else:
                    img.save(buf, format="JPEG")
                raw = buf.getvalue()
                info = tarfile.TarInfo(name=self._member_name(i))
                info.size = len(raw)
                tf.addfile(info, BytesIO(raw))
