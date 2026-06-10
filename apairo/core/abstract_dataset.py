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

    Subclasses must implement ``__len__`` and ``_load``.
    ``_load(idx)`` must return a :class:`~apairo.core.sample.Sample` with raw data
    (no transforms applied).  ``__getitem__``, ``__iter__``, and ``__next__`` are
    provided by this base class: ``__getitem__`` applies registered transforms on
    top of ``_load``; iteration uses index-based access over ``__len__``.

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

    def __iter__(self):
        self._iter_pos = 0
        return self

    def __next__(self):
        if self._iter_pos >= len(self):
            raise StopIteration
        sample = self[self._iter_pos]
        self._iter_pos += 1
        return sample

    def transform(
        self,
        key_or_fn,
        fn: Callable | None = None,
        output: str | None = None,
        keep: bool = True,
    ) -> "AbstractDataset":
        """Register a transform in the pipeline, applied at access time.

        Two forms:

        **Per-channel** -- ``transform(key, fn, output=None, keep=True)``

        ``fn`` receives ``sample.data[key]`` and returns the transformed value.
        By default the result overwrites ``key``.  Pass ``output`` to publish it
        as a new channel while leaving ``key`` untouched::

            ds.transform("lidar", RangeFilter(max=50), output="lidar_f")
            ds.transform("lidar_f", Normalize())          # reads published channel
            ds.transform("lidar_f", Voxelize())           # same source, different branch

        Set ``keep=False`` together with ``output`` to drop the published channel
        from the final sample (useful for intermediate results)::

            ds.transform("lidar", compute_mask, output="_mask", keep=False)
            ds.transform(lambda s: apply_mask(s, "_mask"))

        **Sample-level** -- ``transform(fn)``

        ``fn`` receives and returns the full :class:`~apairo.core.sample.Sample`.
        Use this when an operation must touch several channels consistently::

            def range_filter(sample):
                mask = sample.data["lidar"][:, :3].max(axis=1) < 50
                sample.data["lidar"]  = sample.data["lidar"][mask]
                sample.data["labels"] = sample.data["labels"][mask]
                return sample

            ds.transform(range_filter)

        Both forms compose in registration order and return ``self`` for chaining.
        """
        if fn is None:
            step = key_or_fn
        else:
            key = key_or_fn
            def step(sample: Sample, _key=key, _fn=fn, _out=output) -> Sample:
                if _key in sample.data:
                    result = _fn(sample.data[_key])
                    sample.data[_out if _out is not None else _key] = result
                return sample

        if not hasattr(self, "_pipeline"):
            self._pipeline: list[Callable] = []
        self._pipeline.append(step)

        if output is not None and not keep:
            if not hasattr(self, "_drop_keys"):
                self._drop_keys: set[str] = set()
            self._drop_keys.add(output)

        return self

    def _apply_transforms(self, sample: Sample) -> Sample:
        for fn in getattr(self, "_pipeline", []):
            sample = fn(sample)
        for key in getattr(self, "_drop_keys", set()):
            sample.data.pop(key, None)
        return sample

    def select(self, keys: list[str]) -> "AbstractDataset":
        """Return a view of this dataset restricted to *keys*.

        Calls ``self[idx]`` (transforms applied) then projects to the requested
        channels.  The primary use case is narrowing scope before caching::

            ds.transform("ground_height_csf", expensive_smooth)
            ds_prior = ds.select(["ground_height_csf"]).cache()

        Returns:
            :class:`~apairo.core.channel_view.ChannelView`
        """
        from apairo.core.channel_view import ChannelView
        return ChannelView(self, keys)

    def cache(self) -> "AbstractDataset":
        """Materialise all samples into RAM and return a cached dataset.

        The full dataset is iterated once at call time; all subsequent accesses
        are served from memory with no I/O.  Use after ``.filter()`` or
        ``.select()`` to keep the memory footprint manageable::

            ds_prior = ds.select(["ground_height_csf"]).cache()

        .. warning::
            All samples are loaded into RAM.  Ensure the dataset fits in memory
            before calling.

        Returns:
            :class:`~apairo.core.cached_dataset.CachedDataset`
        """
        from apairo.core.cached_dataset import CachedDataset
        return CachedDataset(self)

    def concat(self, *others: "AbstractDataset") -> "AbstractDataset":
        """Concatenate this dataset with *others* along the frame axis.

        Sugar for ``ConcatDataset([self, *others])``.  Symmetric counterpart
        to :meth:`join`, which merges along the channel axis::

            # frame axis  — more samples, same channels
            combined = ds_kitti.concat(ds_goose)

            # channel axis — same samples, more channels
            combined = ds_base.join(ds_prior)

        Returns:
            :class:`~apairo.dataset.concat.ConcatDataset`
        """
        from apairo.dataset.concat import ConcatDataset
        return ConcatDataset([self, *others])

    def repeat(self, n: int) -> "AbstractDataset":
        """Repeat this dataset *n* times along the frame axis.

        Sugar for ``ConcatDataset([self] * n)``.  With stochastic transforms
        each copy produces independently-augmented samples, effectively giving
        *n* times more gradient updates per epoch::

            ds_aug = ds_train.transform(SparseAugment(...)).repeat(4)

        Args:
            n: Number of repetitions (must be >= 1).

        Returns:
            :class:`~apairo.dataset.concat.ConcatDataset`
        """
        from apairo.dataset.concat import ConcatDataset
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"n must be a positive integer, got {n!r}")
        return ConcatDataset([self] * n)

    def join(self, *others: "AbstractDataset", on_collision: str = "raise") -> "AbstractDataset":
        """Merge channels from this dataset and *others* into a single dataset.

        Sugar for ``ZipDataset(self, *others)``.  All datasets must have the
        same length.  Transforms registered on each parent are applied before
        merging::

            combined = ds_base.join(ds_prior)
            combined[0].data  # union of both datasets' channels

        Args:
            others: One or more datasets of the same length as ``self``.
            on_collision: ``"raise"`` (default) or ``"last"``.

        Returns:
            :class:`~apairo.dataset.zip.ZipDataset`
        """
        from apairo.dataset.zip import ZipDataset
        return ZipDataset(self, *others, on_collision=on_collision)

    def filter_sequences(self, seq_ids) -> "AbstractDataset":
        """Return a FilteredView restricted to frames from *seq_ids*.

        Requires ``frame_sequence_ids`` to be available on this dataset
        (provided by :class:`~apairo.core.profiled_dataset.ProfiledDataset`
        and :class:`~apairo.core.filtered_view.FilteredView`)::

            ds_train = ds_filtered.filter_sequences(train_seqs)
            ds_val   = ds_filtered.filter_sequences([val_seq])
        """
        import numpy as np
        ids = self.frame_sequence_ids
        return self.filter(np.where(np.isin(ids, seq_ids))[0])

    def filter(
        self,
        key_or_fn_or_indices,
        fn: Callable | None = None,
    ) -> "AbstractDataset":
        """Return a filtered view of this dataset.

        Three forms:

        **Pre-computed indices** -- ``filter(indices)``

        Pass a previously saved index array directly — no sweep::

            np.save("valid.npy", view.indices)
            # later:
            view = ds.filter(np.load("valid.npy"))

        **Sample-level** -- ``filter(fn)``

        ``fn`` receives the full :class:`~apairo.core.sample.Sample` (transforms
        applied) and returns ``True`` to keep the frame.  Sweeps the full
        dataset once::

            ds.filter(lambda s: s.data["lidar"].shape[0] > 100)

        **Per-channel** -- ``filter(key, fn)``

        ``fn`` receives ``sample.data[key]`` (raw, before transforms) and returns
        ``True`` to keep the frame.  Only the specified channel is loaded during
        the sweep::

            ds.filter("trav_gt", lambda gt: (gt == 1).sum() >= 50)

        Returns:
            :class:`~apairo.core.filtered_view.FilteredView`
        """
        import numpy as np
        from apairo.core.filtered_view import FilteredView

        if isinstance(key_or_fn_or_indices, (np.ndarray, list)):
            return FilteredView(self, key_or_fn_or_indices)

        if fn is None:
            indices = [i for i in range(len(self)) if key_or_fn_or_indices(self[i])]
        else:
            key = key_or_fn_or_indices
            indices = [i for i in range(len(self)) if fn(self._load(i).data[key])]

        return FilteredView(self, indices)

    def load(self, key: str, idx: int):
        return self.loaders[key][idx]

    @abstractmethod
    def _load(self, idx: int) -> "Sample": ...

    def __getitem__(self, idx: int) -> "Sample":
        return self._apply_transforms(self._load(idx))

    def __len__(self) -> int: ...
