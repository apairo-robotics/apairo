from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor



class Transform(ABC):

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Tensor:
        ...
