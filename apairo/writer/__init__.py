from .npy_writer import NPYWriter
from .bin_writer import BINWriter

WRITERS: dict[str, type] = {
    "npy": NPYWriter,
    "npys": NPYWriter,
    "bin": BINWriter,
}

__all__ = ["NPYWriter", "BINWriter", "WRITERS"]
