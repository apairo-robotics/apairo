from __future__ import annotations

import tarfile
from collections.abc import Callable
from io import BytesIO
from pathlib import Path

import numpy as np

from apairo.core.abstract_loader import AbstractLoader


class TarImageLoader(AbstractLoader):
    """Loader for images packed in a tar archive.

    The loader only handles tar mechanics (listing, extraction, decoding);
    it has no naming convention of its own.  The *dataset* imposes one by
    passing ``name_to_index``, mapping each member name to a frame index
    (return ``None`` to skip a member).  Reusable policies live in
    :mod:`apairo.utils.naming`.

    Args:
        tar_path: Path to the tar file (e.g. ``images.tar``).
        n_frames: Total number of frames (length of the dataset).
        name_to_index: Member name -> frame index policy, imposed by the
            dataset (e.g. :func:`apairo.utils.naming.integer_frame_index`).

    Note:
        The tar file is opened and closed for every ``__getitem__`` call.
        For high-throughput training, consider extracting images to disk first.
    """

    def __init__(
        self,
        tar_path: str | Path,
        n_frames: int,
        name_to_index: Callable[[str], int | None],
    ) -> None:
        self._tar_path = str(Path(tar_path))
        self._n_frames = n_frames
        self._name_to_index = name_to_index
        self._index: dict[int, str] | None = None

    def _ensure_index(self) -> dict[int, str]:
        if self._index is None:
            idx: dict[int, str] = {}
            with tarfile.open(self._tar_path, mode="r") as tf:
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    i = self._name_to_index(m.name)
                    if i is not None and i not in idx:
                        idx[i] = m.name
            self._index = idx
        return self._index

    def __len__(self) -> int:
        return self._n_frames

    def member_name(self, idx: int) -> str | None:
        """Tar member name for frame *idx*, or ``None`` if absent.

        Useful for byte-level copies (re-packing frames without a decode /
        re-encode round trip).
        """
        return self._ensure_index().get(int(idx))

    def __getitem__(self, idx: int) -> np.ndarray:
        member_name = self._ensure_index().get(int(idx))
        if member_name is None:
            raise FileNotFoundError(f"Image index {idx} not found in {self._tar_path}")
        with tarfile.open(self._tar_path, mode="r") as tf:
            fobj = tf.extractfile(member_name)
            if fobj is None:
                raise FileNotFoundError(
                    f"Member '{member_name}' not found in {self._tar_path}"
                )
            return self._decode(fobj.read())

    @staticmethod
    def _decode(data: bytes) -> np.ndarray:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "Pillow is required for TarImageLoader. "
                "Install with: pip install Pillow"
            ) from exc
        img: Image.Image = Image.open(BytesIO(data))
        if img.mode in ("P", "CMYK", "YCbCr", "I", "F"):
            img = img.convert("RGB")
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = arr[..., None]
        return arr

    @property
    def shape(self) -> tuple[int, ...]:
        return (0, 0, 3)
