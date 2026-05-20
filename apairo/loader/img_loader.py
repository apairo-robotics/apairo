import os
from typing import List, Tuple
from torchvision.io import read_image
from ..core import AbstractLoader


class IMGLoader(AbstractLoader):
    r"""A :class:`Loader` class for images in a directory.

    It uses the :meth:`torchvision.io.read_image` function to read the images
    and so supports all the formats that the function supports : `png` and `jpg`.

    Args:
        directory (str) :
            The directory that contains the images.

    Notes:
        Actually, the loader stock all the images names in memory.
        It can be change if it takes too much memory.
    """

    directory: str
    files: List[str]
    index: int
    shape: Tuple[int, ...]

    def __init__(self, directory):

        self.directory = directory
        self.files = list(sorted(
            filter(lambda f: f[-3:] in {"png", "jpg"}, os.listdir(directory)),
            key=lambda f: int(f.split(".")[0])
        ))
        self.index = 0
        self._shape = read_image(os.path.join(self.directory, self.files[0])).shape

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return read_image(os.path.join(self.directory, self.files[idx]))

    @property
    def shape(self):
        return self._shape
