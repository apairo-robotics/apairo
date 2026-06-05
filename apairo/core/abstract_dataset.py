from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
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
from .sample import Sample


class AbstractDataset(ABC):
    """Base class for all robot datasets.

    Subclasses must implement ``__len__``, ``__getitem__``, ``__iter__``, ``__next__``.
    ``__getitem__(idx)`` must return a :class:`~apairo.core.sample.Sample`.

    Attributes:
        available_keys: Frozenset of channel names this dataset type can provide.
        keys: Active channels loaded for this instance.
        timestamps: Per-channel timestamp arrays, or ``None`` for synchronous datasets.
        loaders: Per-channel loader objects.
        calibration: Sensor extrinsics -- see :attr:`calibration`.
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

        Keys follow the convention ``"<from>_to_<to>"`` and values are 4x4
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

    def transform(self, key: str, fn: Callable) -> "AbstractDataset":
        """Register a transform applied to *key* at access time.

        Multiple calls on the same key compose in order (first registered,
        first applied).  Returns ``self`` for chaining::

            ds.transform("poses", To4x4()) \\
              .transform("lidar", RangeFilter(max=50)) \\
              .transform("lidar", Normalize())
        """
        if not hasattr(self, "_transforms"):
            self._transforms: dict[str, list[Callable]] = {}
        self._transforms.setdefault(key, []).append(fn)
        return self

    def sample_transform(self, fn: Callable[["Sample"], "Sample"]) -> "AbstractDataset":
        """Register a transform applied to the full :class:`~apairo.core.sample.Sample` at access time.

        Use this when an operation must touch several channels consistently
        (e.g. a range filter that must keep the same points in both ``lidar``
        and ``labels``).  The callable receives and returns a
        :class:`~apairo.core.sample.Sample`.

        Multiple calls compose in registration order.  Returns ``self`` for
        chaining::

            def sync_filter(sample):
                mask = sample.data["lidar"][:, 0] < 50
                sample.data["lidar"]  = sample.data["lidar"][mask]
                sample.data["labels"] = sample.data["labels"][mask]
                return sample

            ds.sample_transform(sync_filter)
        """
        if not hasattr(self, "_sample_transforms"):
            self._sample_transforms: list[Callable] = []
        self._sample_transforms.append(fn)
        return self

    def _apply_transforms(self, sample: Sample) -> Sample:
        for key, fns in getattr(self, "_transforms", {}).items():
            if key in sample.data:
                val = sample.data[key]
                for fn in fns:
                    val = fn(val)
                sample.data[key] = val
        for fn in getattr(self, "_sample_transforms", []):
            sample = fn(sample)
        return sample

    def load(self, key: str, idx: int):
        return self.loaders[key][idx]

    @abstractmethod
    def _load(self, idx: int) -> "Sample": ...

    def __getitem__(self, idx: int) -> "Sample":
        return self._apply_transforms(self._load(idx))

    def __len__(self) -> int: ...
