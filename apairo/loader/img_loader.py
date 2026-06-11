import os
import numpy as np
from typing import List, Tuple

from apairo.core import AbstractLoader


class IMGLoader(AbstractLoader):
    r"""A :class:`Loader` class for images in a directory.

    Uses Pillow to read images, returning ``np.ndarray`` of shape (H, W, C) uint8.
    Supports PNG, JPG and any format Pillow supports.

    File naming is a dataset concern: pass ``files`` (ordered by frame
    index) to impose the dataset's convention.  When omitted, the legacy
    default applies — ``<int>.{png,jpg}`` members sorted numerically.

    Args:
        directory (str) :
            The directory that contains the images.
        files (list[str], optional) :
            Frame-ordered file names resolved by the dataset.
    """

    directory: str
    files: List[str]
    shape: Tuple[int, ...]

    def __init__(self, directory, files: List[str] | None = None):
        try:
            from PIL import Image as _Image  # noqa: F401
        except ImportError:
            raise ImportError(
                "Image loading requires Pillow. " "Install it with: pip install Pillow"
            )
        self.directory = directory
        self.files = list(files) if files is not None else list(
            sorted(
                filter(lambda f: f[-3:] in {"png", "jpg"}, os.listdir(directory)),
                key=lambda f: int(f.split(".")[0]),
            )
        )
        from PIL import Image

        with Image.open(os.path.join(self.directory, self.files[0])) as img:
            w, h = img.size
            n = len(img.getbands())
        self._shape = (h, w, n) if n > 1 else (h, w)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> np.ndarray:
        from PIL import Image

        path = os.path.join(self.directory, self.files[idx])
        return np.array(Image.open(path))

    @property
    def shape(self):
        return self._shape
