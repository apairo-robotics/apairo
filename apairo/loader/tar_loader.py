from __future__ import annotations
import os
import re
import tarfile
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

from apairo.core.abstract_loader import AbstractLoader


class TarImageLoader(AbstractLoader):
    """Loader for images packed in a tar archive, named by integer frame index.

    The archive is expected to contain files like ``0.jpg``, ``1.jpg``, …
    (zero-padded variants such as ``000000.jpg`` are also accepted).  The
    member index is built lazily on the first access.

    Args:
        tar_path: Path to the ``images.tar`` file.
        n_frames: Total number of frames (length of the dataset).

    Note:
        The tar file is opened and closed for every ``__getitem__`` call.
        For high-throughput training, consider extracting images to disk first.
    """

    def __init__(self, tar_path: str | Path, n_frames: int) -> None:
        self._tar_path = str(Path(tar_path))
        self._n_frames = n_frames
        self._index: Optional[dict[int, str]] = None

    def _ensure_index(self) -> dict[int, str]:
        if self._index is None:
            idx: dict[int, str] = {}
            with tarfile.open(self._tar_path, mode="r") as tf:
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    base = os.path.basename(m.name)
                    match = re.match(r"^0*([0-9]+)\.(jpe?g|png)$", base, re.IGNORECASE)
                    if match:
                        i = int(match.group(1))
                        if i not in idx:
                            idx[i] = m.name
            self._index = idx
        return self._index

    def __len__(self) -> int:
        return self._n_frames

    def __getitem__(self, idx: int) -> np.ndarray:
        index = self._ensure_index()
        member_name = index.get(int(idx))
        with tarfile.open(self._tar_path, mode="r") as tf:
            if member_name is not None:
                fobj = tf.extractfile(member_name)
                if fobj is None:
                    raise FileNotFoundError(
                        f"Member '{member_name}' not found in {self._tar_path}"
                    )
                return self._decode(fobj.read())
            for pat in (
                f"{idx}.jpg",
                f"{idx}.jpeg",
                f"{idx:04d}.jpg",
                f"{idx:06d}.jpg",
            ):
                try:
                    fobj = tf.extractfile(pat)
                    if fobj is not None:
                        return self._decode(fobj.read())
                except KeyError:
                    pass
            raise FileNotFoundError(
                f"Image index {idx} not found in {self._tar_path}"
            )

    @staticmethod
    def _decode(data: bytes) -> np.ndarray:
        try:
            from PIL import Image
        except ImportError:
            raise ImportError(
                "Pillow is required for TarImageLoader. "
                "Install with: pip install Pillow"
            )
        img = Image.open(BytesIO(data))
        if img.mode in ("P", "CMYK", "YCbCr", "I", "F"):
            img = img.convert("RGB")
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = arr[..., None]
        return arr

    @property
    def shape(self) -> Tuple[int, ...]:
        return (0, 0, 3)
