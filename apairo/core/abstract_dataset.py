import copy
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Literal,
    NamedTuple,
)

import numpy as np

if TYPE_CHECKING:
    from apairo.core.filtered_view import FilteredView
    from apairo.core.synchronized_view import ChannelStrategy
from . import abstract_loader
from .config import Calibration, read_calibration
from .sample import Sample
from .utils.exceptions import KeysDuplicateError, KeysEmptyError
from .utils.typing import _Key


class FrameRef(NamedTuple):
    """Where a global frame index comes from -- for layout-aware tooling.

    Returned by :meth:`AbstractDataset.frame_info`. Lets a visualizer or splitter
    map a flat index back to its origin without reaching into private timeline
    state.

    Attributes:
        sequence: Sub-sequence the frame belongs to (``None`` for a single,
            unnamed sequence).
        channel: Channel that produced the event -- asynchronous datasets
            interleave channels, so one event is one channel. ``None`` for a
            synchronous frame, which is *all* channels at the same row.
        row: Frame index within that channel/sequence.
    """

    sequence: str | None
    channel: str | None
    row: int


class _ChannelStep:
    """Pipeline step for ``transform(key, fn)`` -- module-level (not a closure)
    so a transformed dataset stays picklable for spawn-based DataLoader workers."""

    __slots__ = ("key", "fn", "output")

    def __init__(self, key: str, fn: Callable, output: str | None) -> None:
        self.key, self.fn, self.output = key, fn, output

    def __call__(self, sample: Sample) -> Sample:
        if self.key in sample.data:
            result = self.fn(sample.data[self.key])
            sample.data[self.output if self.output is not None else self.key] = result
        return sample


class _PreprocessorStep:
    """Pipeline step for ``transform(preprocessor)`` -- lazy preview publishing
    the result under *output* (single-output, renameable) or under every
    declared ``output_keys`` (multi-output). Module-level for picklability,
    like _ChannelStep."""

    __slots__ = ("preprocessor", "output")

    def __init__(self, preprocessor, output: str | None) -> None:
        self.preprocessor, self.output = preprocessor, output

    def __call__(self, sample: Sample) -> Sample:
        from apairo.core.preprocessor import as_output_dict

        missing = [k for k in self.preprocessor.input_keys if k not in sample.data]
        if missing:
            raise KeyError(
                f"{type(self.preprocessor).__name__} needs input channels "
                f"{missing} absent from the sample (available: "
                f"{sorted(sample.data)})."
            )
        result = self.preprocessor(sample)
        if self.output is not None:
            sample.data[self.output] = result
        else:
            sample.data.update(as_output_dict(self.preprocessor, result))
        return sample


class AbstractDataset(ABC):
    """Base class for all robot datasets.

    Subclasses must implement ``__len__`` and ``_load``.
    ``_load(idx)`` must return a :class:`~apairo.core.sample.Sample` with raw data
    (no transforms applied).  ``__getitem__`` and ``__iter__`` are provided by
    this base class: ``__getitem__`` applies registered transforms on top of
    ``_load``; iteration uses index-based access over ``__len__``.

    Attributes:
        available_keys: Frozenset of channel names this dataset type can provide.
        keys: Active channels loaded for this instance.
        timestamps: Per-channel timestamp arrays, or ``None`` for synchronous datasets.
        loaders: Per-channel loader objects.
        calibration: Sensor extrinsics -- see :attr:`calibration`.
    """

    available_keys: ClassVar[frozenset[str]] = frozenset()
    """Channels this dataset type can provide.  Override in each concrete class."""

    timestamps: dict | None
    loaders: dict[_Key, abstract_loader.AbstractLoader]
    synchronous: bool
    profile: dict[_Key, str] | None
    # Lazily created by transform(); read via getattr with a default everywhere.
    _pipeline: list[Callable[[Sample], Sample]]
    _drop_keys: set[str]

    def _set_keys(self, keys: list[_Key]) -> None:
        if len(keys) == 0:
            raise KeysEmptyError
        if len(set(keys)) != len(keys):
            raise KeysDuplicateError
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
    def calibration(self) -> Calibration:
        """Sensor extrinsics for this dataset, as a :class:`~apairo.core.config.Calibration`.

        Keys follow ``"<parent>_to_<child>"`` and values are 4x4 float64 matrices.
        Resolve any pair with ``ds.calibration.get_tf(source, target)``.

        Read from ``<root_dir>/.apairo/calibration.yaml`` -- every dataset with a
        ``root_dir`` exposes its extrinsics, not just :class:`RawDataset`. Empty when
        no calibration file exists, or the dataset has no on-disk root (e.g. a
        cached or concatenated view). The async family overrides this to merge
        per-sequence tables (see :class:`~apairo.core.root_sequence.RootSequenceMixin`).
        """
        root = getattr(self, "root_dir", None)
        return read_calibration(root) if root is not None else Calibration()

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def transform(
        self,
        key_or_fn,
        fn: Callable | None = None,
        output: str | None = None,
        keep: bool = True,
        in_place: bool = True,
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

        **Preprocessor** -- ``transform(preprocessor)``

        A :class:`~apairo.core.preprocessor.FramePreprocessor` runs lazily:
        its result is published under its declared ``output_key`` (override
        with ``output=``) at access time, nothing is written to disk.  This
        is how to preview a preprocess before materializing it::

            p = TravLabel()
            preview = ds.transform(p)        # lazy, nothing written
            preview[42].data["trav_label"]   # inspect, iterate on p
            ds.run_preprocess(p)             # materialize once satisfied

        A multi-output preprocessor (``output_keys``) publishes every key of
        its returned dict; ``output=`` cannot rename it.

        A :class:`~apairo.core.preprocessor.SequencePreprocessor` is rejected:
        it needs the full sequence at once, materialization is the point.

        All forms compose in registration order and return ``self`` for
        chaining.

        **Branching** -- ``transform(..., in_place=False)``

        Pass ``in_place=False`` to leave ``self`` untouched and register the
        transform on an independent branch instead: a lightweight copy sharing
        loaders and indices but owning its pipeline (transforms already
        registered on ``self`` are inherited)::

            base = Rellis3DDataset(root, keys=["lidar"])
            v1 = base.transform(augment_v1, in_place=False)
            v2 = base.transform(augment_v2, in_place=False)   # independent

        .. warning::
            By default transforms are registered **in place**: the return value
            is the same object, so ``v1 = ds.transform(a)`` and
            ``v2 = ds.transform(b)`` leave ``v1 is v2 is ds`` with *both*
            transforms stacked.  To build independent variants, pass
            ``in_place=False`` (or branch first, e.g. ``ds.filter(...)``).
        """
        step: Callable[[Sample], Sample]
        drop_key: str | list[str] | None = output
        if fn is None:
            from apairo.core.preprocessor import (
                FramePreprocessor,
                Preprocessor,
                SequencePreprocessor,
            )

            if isinstance(key_or_fn, type) and issubclass(key_or_fn, Preprocessor):
                raise TypeError(
                    f"transform() expects a preprocessor instance, not the "
                    f"class -- did you mean transform({key_or_fn.__name__}())?"
                )
            if isinstance(key_or_fn, SequencePreprocessor):
                raise TypeError(
                    f"{type(key_or_fn).__name__} is a SequencePreprocessor: "
                    f"it needs the full sequence at once and cannot run "
                    f"lazily. Materialize it with run_preprocess() instead."
                )
            if isinstance(key_or_fn, FramePreprocessor):
                if output is not None and key_or_fn.output_keys is not None:
                    raise TypeError(
                        f"output= cannot rename the multi-output "
                        f"{type(key_or_fn).__name__} (output_keys="
                        f"{key_or_fn.output_keys})."
                    )
                drop_key = [output] if output is not None else key_or_fn.outputs
                step = _PreprocessorStep(key_or_fn, output)
            elif not callable(key_or_fn):
                raise TypeError(
                    f"transform(fn) expects a callable sample->sample or a "
                    f"FramePreprocessor; got {key_or_fn!r}. For a "
                    f"per-channel transform use transform(key, fn)."
                )
            else:
                step = key_or_fn
        else:
            key = key_or_fn
            if not isinstance(key, str) or not callable(fn):
                raise TypeError(
                    f"transform(key, fn) expects a channel name then a callable; "
                    f"got transform({key_or_fn!r}, {fn!r}). Arguments reversed?"
                )
            step = _ChannelStep(key, fn, output)

        target = self if in_place else self._branch()

        if not hasattr(target, "_pipeline"):
            target._pipeline = []
        target._pipeline.append(step)

        if drop_key is not None and not keep:
            if not hasattr(target, "_drop_keys"):
                target._drop_keys = set()
            keys = drop_key if isinstance(drop_key, list) else [drop_key]
            target._drop_keys.update(keys)

        return target

    def _branch(self) -> "AbstractDataset":
        """Independent branch of this dataset: a shallow copy sharing loaders
        and indices but owning its transform pipeline."""
        clone = copy.copy(self)
        clone._pipeline = list(getattr(self, "_pipeline", []))
        clone._drop_keys = set(getattr(self, "_drop_keys", set()))
        return clone

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

            ds.transform("lidar", expensive_ground_prior, output="ground_prior")
            ds_prior = ds.select(["ground_prior"]).cache()

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

            ds_prior = ds.select(["ground_prior"]).cache()

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

    def join(
        self,
        *others: "AbstractDataset",
        on_collision: 'Literal["raise", "last"]' = "raise",
    ) -> "AbstractDataset":
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

    def filter_sequences(self, seq_ids) -> "FilteredView":
        """Return a FilteredView restricted to frames from *seq_ids*.

        Requires ``frame_sequence_ids`` to be available on this dataset
        (provided by :class:`~apairo.core.profiled_dataset.ProfiledDataset`
        and :class:`~apairo.core.filtered_view.FilteredView`)::

            ds_train = ds_filtered.filter_sequences(train_seqs)
            ds_val   = ds_filtered.filter_sequences([val_seq])
        """
        ids = self.frame_sequence_ids
        return self.filter(np.where(np.isin(ids, seq_ids))[0])

    def filter(
        self,
        key_or_fn_or_indices,
        fn: Callable | None = None,
    ) -> "FilteredView":
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

        ``fn`` receives the channel value and returns ``True`` to keep the
        frame.  When the dataset exposes per-frame loaders, only the specified
        channel is read during the sweep (raw, before transforms); views
        without loaders fall back to loading the full sample::

            ds.filter("trav_gt", lambda gt: (gt == 1).sum() >= 50)

        Only available on synchronous datasets -- in an asynchronous timeline
        each index holds a single channel, so a per-channel sweep is undefined.
        Call :meth:`synchronize` first.

        Returns:
            :class:`~apairo.core.filtered_view.FilteredView`
        """
        from apairo.core.filtered_view import FilteredView

        if isinstance(key_or_fn_or_indices, (np.ndarray, list)):
            return FilteredView(self, key_or_fn_or_indices)

        if fn is None:
            indices = [i for i in range(len(self)) if key_or_fn_or_indices(self[i])]
        else:
            key = key_or_fn_or_indices
            if not self.is_synchronous:
                raise ValueError(
                    "Per-channel filter is undefined on an asynchronous dataset: "
                    "each timeline index holds a single channel. Call "
                    ".synchronize() first, or use the sample-level form "
                    "filter(fn)."
                )
            loaders = getattr(self, "loaders", None)
            if loaders and key in loaders:
                indices = [i for i in range(len(self)) if fn(loaders[key][i])]
            else:
                indices = [i for i in range(len(self)) if fn(self._load(i).data[key])]

        return FilteredView(self, indices)

    def window(
        self,
        size: int,
        stride: int = 1,
        reduce: Callable | None = None,
        boundary: str = "clip",
    ) -> "AbstractDataset":
        """Group each frame with its temporal neighbours, then reduce to one sample.

        Returns a view whose frame ``j`` is the reduction of a causal window
        ending at an *anchor* frame: the anchor plus the ``size - 1`` preceding
        frames spaced ``stride`` apart, ordered oldest -> newest. Membership is
        index arithmetic computed at construction; the window's samples are read
        lazily and handed to ``reduce`` at access time.

        This is the random-access counterpart to the stateful
        ``AccumulateFrames`` transform: windows are addressed by index, so the
        result stays correct under ``.split()``, shuffling and multi-worker
        ``DataLoader``. It works on any ordered dataset -- a natively synchronous
        one, or a synchronous view from :meth:`synchronize`::

            ds = (TartanKittiDataset(seq, keys=["lidar", "pose"])
                    .synchronize(reference="lidar", method={"pose": Se3Interp()})
                    .window(size=5, stride=1, reduce=StackByPose()))

        Windows never cross a sequence boundary (``frame_sequence_ids``); when the
        parent has none, the whole dataset is treated as one sequence.

        Args:
            size: Frames per window, including the anchor (``>= 1``).
            stride: Gap, in frames, between two window members.
            reduce: Callable ``list[Sample] -> Sample``. **Required** -- a
                windowed frame holds several samples and is not a valid sample
                until reduced (see ``StackByPose`` in ``apairo_transform``).
            boundary: ``"clip"`` -- shorter windows near a sequence start, one
                output per frame; ``"drop"`` -- keep only full windows.

        Returns:
            :class:`~apairo.core.window_view.WindowView`
        """
        from apairo.core.window_view import WindowView

        return WindowView(self, size, stride, reduce, boundary)

    def synchronize(
        self,
        reference: "str | np.ndarray | None" = None,
        method: "ChannelStrategy | dict[str, ChannelStrategy]" = "previous",
        tolerance: float | None = None,
    ) -> "AbstractDataset":
        """Resample this asynchronous dataset onto a single reference clock.

        Returns a synchronous view where index ``i`` is the *i*-th tick of the
        reference clock, with every channel matched by timestamp.  The result
        behaves like any synchronous dataset: complete samples, random access,
        and the full chaining API (``filter``, ``select``, ``cache``,
        ``join``, PyTorch ``DataLoader`` with shuffling)::

            ds = TartanKittiDataset(seq, keys=["velodyne_0", "image_left"])
            ds_sync = ds.synchronize(reference="velodyne_0", tolerance=0.05)
            ds_sync[0].data   # {"velodyne_0": ..., "image_left": ...}

        The clock can also be external -- fixed-rate or distance-based::

            # one frame every 100 ms
            ds_10hz = ds.synchronize(reference=np.arange(t0, t1, 0.1))

            # one frame every 0.5 m travelled (from odometry)
            from apairo.utils import clock_from_distance
            clock = clock_from_distance(odom_ts, odom_xy, step=0.5)
            ds_spatial = ds.synchronize(reference=clock)

        Continuous signals (poses, IMU, commands) can be interpolated at the
        reference instant instead of matched, with per-channel strategies::

            from apairo_transform.interp import Se3Interp

            ds_sync = ds.synchronize(
                reference="velodyne_0",
                method={"gicp_poses": Se3Interp()},   # others -> "previous"
            )

        Args:
            reference: Channel name providing the clock; ``None`` for the
                lowest-frequency channel (so every frame sees fresh data); or
                an ascending array of timestamps to use as an external clock.
            method: Strategy for every channel, or a dict of per-channel
                strategies (unlisted channels default to ``"previous"``).
                ``"previous"`` -- last event with ``t <= t_ref`` (zero-order
                hold, online-style; ``"latest"`` is a deprecated alias);
                ``"next"`` -- first event with ``t >= t_ref``; ``"nearest"``
                -- event closest in time, either side (ties favour the
                earlier event); a callable ``(channel_ts, ref_ts) ->
                indices`` implementing a custom matching strategy (negative
                index = no match); or an
                :class:`~apairo.core.interpolator.Interpolator` synthesizing
                the value at ``t_ref`` from the two bracketing events.
            tolerance: Maximum ``|t - t_ref|`` in seconds.  Reference frames
                where any channel has no match within tolerance are dropped
                (for interpolated channels, both bracketing events count).

        Returns:
            :class:`~apairo.core.synchronized_view.SynchronizedView`
        """
        from apairo.core.synchronized_view import SynchronizedView

        return SynchronizedView(
            self, reference=reference, method=method, tolerance=tolerance
        )

    def load(self, key: str, idx: int):
        return self.loaders[key][idx]

    @abstractmethod
    def _load(self, idx: int) -> "Sample": ...

    def __getitem__(self, idx: int) -> "Sample":
        return self._apply_transforms(self._load(idx))

    def frame_info(self, idx: int) -> "FrameRef":
        """Provenance of global frame *idx* as ``FrameRef(sequence, channel, row)``.

        Lets layout-aware tooling (e.g. a visualizer) map a flat index back to the
        channel and frame-within-channel it came from, instead of reaching into
        private timeline state. Read-only.

        Default (synchronous datasets): a frame is *all* channels at row ``idx``
        of its sequence, so ``channel`` is ``None`` and ``row`` is ``idx``. The
        asynchronous family overrides this with the single channel + row each
        interleaved event came from."""
        try:
            sequence = self.frame_sequence_ids[idx]
        except (AttributeError, NotImplementedError, RuntimeError):
            sequence = None
        return FrameRef(sequence=sequence, channel=None, row=int(idx))

    @property
    def frame_sequence_ids(self) -> np.ndarray:
        """Sequence id per global frame, shape ``(len(self),)``.

        Provided by datasets and views with a sequence structure. The base
        raises ``AttributeError`` so ``getattr(ds, "frame_sequence_ids", None)``
        keeps working as an availability probe."""
        raise AttributeError(f"{type(self).__name__} exposes no frame_sequence_ids")

    @property
    def frame_stems(self) -> np.ndarray:
        """Filename stem backing each global frame, shape ``(len(self),)``.

        Provided by on-disk datasets and index views; raises ``AttributeError``
        when unavailable, like :attr:`frame_sequence_ids`."""
        raise AttributeError(f"{type(self).__name__} exposes no frame_stems")

    @property
    def frame_channel_ids(self) -> np.ndarray:
        """Channel that produced each global frame, shape ``(len(self),)``.

        Provided by asynchronous datasets and views over them, where a frame
        is one channel's event. Raises ``AttributeError`` on synchronous data
        (a frame there is *all* channels) and on composite frames (e.g. a
        synchronized view, which has no single origin channel), like
        :attr:`frame_sequence_ids`."""
        raise AttributeError(f"{type(self).__name__} exposes no frame_channel_ids")

    @abstractmethod
    def __len__(self) -> int: ...
