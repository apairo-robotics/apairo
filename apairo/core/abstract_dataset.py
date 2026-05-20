from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Union, overload
from . import abstract_loader

from .utils.typing import Timestamp, _Key
from .utils.exceptions import KeysEmptyWarning, KeysDuplicateWarning


class AbstractDataset(ABC):
    r""" :class:`AbstractDataset` is an abstract class for dataset from robots data.

    Each dataset that will be used will have a corresponding class that will inherit from this class.

    The subclass of :class:`AbstractDataset` have to implement the :meth:`__getitem__` method overload for index in the :type:`dict` type.
    This have to be in the timestamp order, ie the timestamp at the index i will be lower than the timestamp at the index i+1.
    Furthermore, :meth:`__getitem__` have to return a :type:`dict` with this format:
    ```python
    { "key": key, "data": data, "timestamp": timestamp }
    ```
    where:
    - key (string)
    - data (torch.Tensor)
    - timestamp (float) # We will have to specify float32 or float64 or double ...

    .. note::
        During the initialization we assume that the timestamps of the data need to be ordered. The :meth:`init_timeline`
        method initalize the :attr:`timeline` attribute with the timestamps of the data.

    Attributes:
        keys (list) :
            The keys that are present in the dataset that will be used (keys belonging to the profile if is present)
        timestamps (Timestamp dict) :
            A dictionary with the timestamps for each key (see the timestamps.py file for more information)
        dataloaders (dict):
            A dictionary with the dataloaders (See the loaders.py file for more information)
        synchronous (bool):
            A bool parameter that indicate if the timestamps of the dara are synchronous
        profile (Optional path) :
            A loaded yaml file that give for each key the type of the data. (ex: npy, multiple npy or image for the time being)

    """

    keys: Union[List[_Key], Sequence[_Key]]
    timestamps: dict | None
    loaders: Dict[_Key, abstract_loader.AbstractLoader]
    synchronous: bool
    profile: Optional[Dict[_Key, str]]

    def _set_keys(self, keys: list[_Key]) -> None:
        if len(keys) == 0:
            raise KeysEmptyWarning
        if len(set(keys)) != len(keys):
            raise KeysDuplicateWarning
        self._keys = keys

    @property
    def keys(self) -> list[_Key]:
        return self._keys

    @keys.setter
    def keys(self, keys: list[_Key]) -> None:
        self._set_keys(keys)

    @property
    def is_synchronous(self) -> bool:
        """True if this dataset has no timestamps (synchronous frame access)."""
        return getattr(self, 'timestamps', None) is None

    @abstractmethod
    def __iter__(self):
        ...

    @abstractmethod
    def __next__(self):
        ...

    def load(self, key: str, idx: int):
        return self.loaders[key][idx]

    @overload
    def __getitem__(self, idx: int) -> Any:
        ...

    @overload
    def __getitem__(self, idx: Dict[str, Sequence[int]]) -> Dict[str, Any]:
        ...

    @overload
    def __getitem__(self, idx: int) -> Any:
        ...

    @abstractmethod
    def __getitem__(
        self,
        idx: Union[int, Dict[str, Sequence[int]]],
    ) -> Union[Any, Dict[str, Any]]:
        ...

    def __len__(self) -> int:
        ...