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

# Imported last: channel_writer pulls the per-frame writer classes defined above.
from .channel_writer import ChannelWriter  # noqa: E402

__all__ = [
    "NPYWriter",
    "BINWriter",
    "ZarrWriter",
    "TarImageWriter",
    "WRITERS",
    "ChannelWriter",
]
