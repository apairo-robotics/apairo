from abc import ABC, abstractmethod
from typing import Union, Tuple
from torch import Tensor
from numpy import ndarray


class AbstractLoader(ABC):
    r"""An interface class representing a :class:`Loader`.

    All subclasses should overwrite :meth:`__getitem__` supporting fetching a
    data sample for a given key. Subclasses could also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the data and :meth:`shape`
    which is expected to return the shape of the data (dimension).

    For example a set of ten images of size 3x224x224 would have a len of 10 and a shape of (3, 224, 224).
    """

    def __init__(self, directory: str, transform=None, *args, **kwargs) -> None:
        ...

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx) -> Union[Tensor, ndarray]:
        pass

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        pass
