from .npy_writer import NPYWriter
from .bin_writer import BINWriter
from .zarr_writer import ZarrWriter
from .tar_writer import TarImageWriter

WRITERS: dict[str, type] = {
    "npy": NPYWriter,
    "npys": NPYWriter,
    "bin": BINWriter,
    "zarr": ZarrWriter,
    "img": TarImageWriter,
}

__all__ = ["NPYWriter", "BINWriter", "ZarrWriter", "TarImageWriter", "WRITERS"]
