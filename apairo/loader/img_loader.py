import os

import numpy as np

from apairo.core import AbstractLoader
from apairo.core.utils.exceptions import FileExtensionError

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def _img_sort_key(name: str) -> tuple[int, object]:
    """Numeric-first sort: a pure-integer stem sorts numerically (0, 1, 2, 10);
    anything else (timestamped/prefixed names) falls back to lexicographic instead
    of crashing the default ``int()`` sort."""
    stem = os.path.splitext(name)[0]
    return (0, int(stem)) if stem.isdigit() else (1, name)


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
    files: list[str]

    def __init__(self, directory, files: list[str] | None = None):
        try:
            from PIL import Image as _Image  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Image loading requires Pillow. Install it with: pip install Pillow"
            ) from exc
        self.directory = directory
        if files is not None:
            self.files = list(files)
        else:
            self.files = sorted(
                (
                    f
                    for f in os.listdir(directory)
                    if os.path.splitext(f)[1].lower() in _IMG_EXTS
                ),
                key=_img_sort_key,
            )
        if not self.files:
            raise FileExtensionError(
                f"No image files ({', '.join(sorted(_IMG_EXTS))}) found in {directory}."
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
