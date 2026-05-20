from __future__ import annotations
from abc import abstractmethod

from apairo.core.abstract_dataset import AbstractDataset
from apairo.core.sample import Sample


class SynchronousDataset(AbstractDataset):
    """Base class for datasets where index i returns a complete synchronous frame.

    All modalities are captured at the same time — no timestamps, no interleaving.
    Compatible with PyTorch's standard DataLoader (random/sequential sampling).

    Subclasses must implement: __len__, __getitem__, __iter__, __next__.
    """

    timestamps = None

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Sample: ...

    @abstractmethod
    def __iter__(self): ...

    @abstractmethod
    def __next__(self) -> Sample: ...
