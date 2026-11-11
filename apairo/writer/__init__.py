from .bin_writer import BINWriter
from .channel_writer import ChannelWriter
from .npy_writer import NPYWriter
from .tar_writer import TarImageWriter
from .zarr_writer import ZarrWriter

WRITERS: dict[str, type] = {
    "npy": NPYWriter,
    "npys": NPYWriter,
    "bin": BINWriter,
    "zarr": ZarrWriter,
    "img": TarImageWriter,
}

__all__ = [
    "NPYWriter",
    "BINWriter",
    "ZarrWriter",
    "TarImageWriter",
    "WRITERS",
    "ChannelWriter",
]
