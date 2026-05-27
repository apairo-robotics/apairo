from abc import ABC, abstractmethod
from typing import (
    Any,
    ClassVar,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Union,
)
import numpy as np
from . import abstract_loader

from .utils.typing import _Key
from .utils.exceptions import KeysEmptyWarning, KeysDuplicateWarning


class AbstractDataset(ABC):
    """Base class for all robot datasets.

    Subclasses must implement ``__len__``, ``__getitem__``, ``__iter__``, ``__next__``.
    ``__getitem__(idx)`` must return a :class:`~apairo.core.sample.Sample`.

    Attributes:
        available_keys: Frozenset of channel names this dataset type can provide.
        keys: Active channels loaded for this instance.
        timestamps: Per-channel timestamp arrays, or ``None`` for synchronous datasets.
        loaders: Per-channel loader objects.
        calibration: Sensor extrinsics — see :attr:`calibration`.
    """

    available_keys: ClassVar[FrozenSet[str]] = frozenset()
    """Channels this dataset type can provide.  Override in each concrete class."""

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
        return getattr(self, "timestamps", None) is None

    @property
    def calibration(self) -> Dict[str, np.ndarray]:
        """Sensor extrinsics for this dataset.

        Keys follow the convention ``"<from>_to_<to>"`` and values are 4×4
        homogeneous transformation matrices (float64).  Returns an empty dict
        when the dataset provides no calibration.

        Override in a subclass to expose the dataset's calibration file::

            @property
            def calibration(self) -> dict[str, np.ndarray]:
                return {"lidar_to_camera": self._load_calib()}
        """
        return {}

    @abstractmethod
    def __iter__(self): ...

    @abstractmethod
    def __next__(self): ...

    def load(self, key: str, idx: int):
        return self.loaders[key][idx]

    @abstractmethod
    def __getitem__(self, idx: int) -> Any: ...

    def __len__(self) -> int: ...
