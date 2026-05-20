from abc import ABC
from typing import Sequence, SupportsFloat, Union


from torch.utils.data import Sampler
from src.utils.types import Timestamp


class AbstractSampler(Sampler, ABC):
    """Every :class:`Sampler` should inherit from this class.

    Such as :class:`Sampler`, every subclass have to provide an implementation of the :meth:`__iter__`, providing a
    way to iterate over indices or lists of indices (batches) of dataset elements,
    and may provide a :meth:`__len__` method that returns the length of the returned iterators.

    Args:
        timestamp (Timestamp) :
            The timestamp of the dataset
        sample_size (int) :
            The size of the batch
        is_uniform (bool) :
            A flag that indicates if the sampler iterate over batches of the same size
    """

    timestamps: Union[Timestamp, Sequence[SupportsFloat]]
    sample_size: int
    is_uniform: bool

    def __init__(self, timestamps: Timestamp, sample_size: int = 1):
        super().__init__()
        self.timestamps = timestamps
        self._sample_size = sample_size

    @property
    def sample_size(self) -> int:
        """Return the batch size"""
        return self._sample_size

    @sample_size.setter
    def sample_size(self, sample_size: int):
        """Set the batch size"""
        if sample_size <= 0:
            raise ValueError("The batch size should be greater than 0")
        self._sample_size = sample_size

    @property
    def keys(self) -> Sequence[str]:
        """Return the keys of the timestamp"""
        return self.timestamps.keys()
