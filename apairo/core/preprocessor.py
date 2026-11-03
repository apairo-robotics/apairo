from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import Any, ClassVar

from apairo.core.sample import Sample


class Preprocessor(ABC):
    """Base class for all apairo preprocessors.

    Subclasses declare their I/O contract as class attributes and implement
    ``__call__`` -- a preprocessor *is* a callable, the same protocol as
    transforms and :class:`~apairo.core.interpolator.Interpolator`.  The same
    instance therefore works in both worlds: lazily in a pipeline
    (``ds.transform(preprocessor)``) or materialized to disk
    (``ds.run_preprocess(preprocessor)``, which reads the class attributes to
    decide how to load inputs, how to save outputs, and how to register the
    new channel in ``.apairo``).

    .. deprecated::
        Implementing the computation as ``process()`` instead of ``__call__``
        still works but is deprecated; a legacy ``process`` is aliased to
        ``__call__`` with a :class:`DeprecationWarning` at class definition.

    Class attributes
    ----------------
    output_key : str
        Subdirectory name for the output channel (e.g. ``"trav_label"``).
    output_keys : list[str]
        Multi-output alternative to ``output_key`` (declaring both is an
        error).  ``__call__`` must then return a ``dict`` with exactly these
        keys; the runner writes one derived channel per key (same
        ``output_loader``) and registers all of them with shared provenance.
        Use this when one computation naturally produces several channels
        (e.g. a voxel structure emitting ``cell_coords`` + ``cell_inv``).
    output_loader : str or None
        The format to store the output in -- any registered format that
        writes: ``"npys"`` (one ``.npy`` per frame), ``"npy"`` (one stacked
        array), ``"img"`` (PNG), ``"zarr"``, ``"csv"``, ``"bin"``, or a
        plugin's. Leave it unset unless the output's nature dictates one: the
        output is then written in its input channel's format when that format
        can hold it, else in ``npys`` / ``npy``. ``run_preprocess(...,
        output_format=...)`` overrides it.
    input_keys : list[str]
        Dataset channels needed as input.
    timestamps_from : str or None
        The source channel whose timestamps this output shares.  Stored in
        ``.apairo`` as provenance.  The runner always writes a
        ``timestamps.txt`` into the output channel's directory.
    sources : list[str] or None
        Provenance -- channels this output was derived from (stored in
        ``.apairo`` for reference).
    """

    # A preprocessor is a callable; concrete subclasses implement __call__
    # (FramePreprocessor and SequencePreprocessor declare it abstract).
    __call__: Callable[..., Any]

    output_key: ClassVar[str]
    output_keys: ClassVar[list[str] | None] = None
    output_loader: ClassVar[str | None] = None
    input_keys: ClassVar[list[str]]
    timestamps_from: ClassVar[str | None] = None
    sources: ClassVar[list[str] | None] = None

    @property
    def outputs(self) -> list[str]:
        """Declared output channel names -- ``output_keys`` or ``[output_key]``."""
        return self.output_keys if self.output_keys is not None else [self.output_key]

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if cls.output_keys is not None:
            if getattr(cls, "output_key", None) is not None:
                raise TypeError(
                    f"{cls.__name__} declares both output_key and output_keys; "
                    f"they are exclusive."
                )
            if not cls.output_keys or len(set(cls.output_keys)) != len(cls.output_keys):
                raise TypeError(
                    f"{cls.__name__}.output_keys must be a non-empty list of "
                    f"unique channel names, got {cls.output_keys!r}."
                )
        legacy = cls.__dict__.get("process")
        if (
            legacy is not None
            and not getattr(legacy, "__isabstractmethod__", False)
            and "__call__" not in cls.__dict__
        ):
            warnings.warn(
                f"{cls.__name__} implements process(); implement __call__ "
                f"instead (process is a deprecated alias).",
                DeprecationWarning,
                stacklevel=3,  # __init_subclass__ <- ABCMeta.__new__ <- class statement
            )
            cls.__call__ = legacy  # type: ignore[method-assign]  # deprecation alias

    def process(self, *args, **kwargs) -> Any:
        """Deprecated alias for calling the instance directly."""
        warnings.warn(
            f"{type(self).__name__}.process() is deprecated; call the "
            f"instance directly (preprocessor(...)).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self(*args, **kwargs)


def as_output_dict(preprocessor: Preprocessor, result: Any) -> dict[str, Any]:
    """Normalize a ``__call__`` result to ``{key: value}`` against the contract.

    Single-output wraps the result under ``output_key``; multi-output requires
    a ``dict`` with exactly the declared ``output_keys``.
    """
    if preprocessor.output_keys is None:
        return {preprocessor.output_key: result}
    if not isinstance(result, dict) or set(result) != set(preprocessor.output_keys):
        got = sorted(result) if isinstance(result, dict) else type(result).__name__
        raise ValueError(
            f"{type(preprocessor).__name__} declares output_keys="
            f"{preprocessor.output_keys} but returned {got}; __call__ must "
            f"return a dict with exactly those keys."
        )
    return result


class FramePreprocessor(Preprocessor):
    """Preprocessor that operates frame-by-frame.

    The runner calls the instance once per input frame.  Use this for
    per-scan operations (label inference, feature extraction, …).

    Output is stored as one file per frame (``000000.npy``, ``000001.npy``,
    …) when ``output_loader`` is ``"npys"`` or ``"bin"``.

    Because a frame preprocessor is just a ``Sample -> value`` callable, it
    can also run lazily -- ``ds.transform(preprocessor)`` publishes its
    result under ``output_key`` at access time, nothing is written.  Preview
    a preprocess this way before materializing it with ``run_preprocess``.

    **Temporal window.** Set ``window_size > 1`` (and optionally
    ``window_stride``) to see each frame with its past neighbours: the frame
    plus the ``window_size - 1`` preceding frames spaced ``window_stride``
    apart, oldest -> newest, clipped at the start of its sequence and never
    crossing into another one -- the semantics of
    :meth:`~apairo.core.abstract_dataset.AbstractDataset.window` with
    ``boundary="clip"``. :meth:`window_reduce` collapses the window into one
    sample, then ``__call__`` runs on it as usual. There is still one output
    per frame, placed and stamped as that (anchor) frame. ``run_preprocess``
    streams: it reads each frame once and keeps only the last
    ``(window_size - 1) * window_stride + 1`` frames of the sequence. It
    refuses a run that selects frames within sequences (a split listing
    frames), where the neighbours would be the selected frames and the
    persisted output would depend on the split; a split of whole sequences
    runs, its windows being those of the whole root. ``window_size``,
    ``window_stride`` and the reducer enter the recipe. Preview lazily with
    ``p.windowed(ds).transform(p)`` -- allowed on a split, since a view
    persists nothing: its windows are over the split's frames.

    Example::

        class TravLabel(FramePreprocessor):
            output_key    = "trav_label"
            output_loader = "npys"
            input_keys    = ["velodyne_0"]
            timestamps_from = "velodyne_0"   # no own timestamps.txt

            def __call__(self, sample: Sample) -> np.ndarray:
                pts = sample.data["velodyne_0"]
                return my_model(pts)

        class BevAccumulated(FramePreprocessor):
            output_key  = "bev_acc"
            input_keys  = ["lidar", "pose"]
            window_size = 10                  # this scan + the 9 before it

            def __init__(self):
                self.window_reduce = StackByPose()   # from apairo_transform

            def __call__(self, sample: Sample) -> np.ndarray:
                return bev_grid(sample.data["lidar"])
    """

    window_size: int = 1
    window_stride: int = 1

    def window_reduce(self, samples: list[Sample]) -> Sample:
        """Collapse a temporal window into the one sample ``__call__`` receives.

        Only used when ``window_size > 1``. *samples* is the causal window
        ending at the anchor frame, oldest -> newest (``samples[-1]`` is the
        anchor), clipped at the start of its sequence. The returned sample must
        carry the anchor's timestamp: the output belongs to the anchor frame.
        It must not mutate *samples*: ``run_preprocess`` hands the same sample
        objects to every window they belong to.

        Override this method, or assign a reducer callable to the instance
        (``self.window_reduce = StackByPose()`` from ``apairo_transform``).
        """
        raise NotImplementedError(
            f"{type(self).__name__} declares window_size={self.window_size} but "
            f"no window_reduce: override window_reduce(samples) -> Sample, or "
            f"assign a reducer such as apairo_transform.StackByPose() to "
            f"self.window_reduce."
        )

    def windowed(self, dataset: Any) -> Any:
        """The view this preprocessor reads frames from: *dataset* itself when
        ``window_size == 1``, else ``dataset.window(window_size, window_stride,
        reduce=window_reduce, boundary="clip")`` -- one reduced sample per frame
        of *dataset*, index for index.

        This is the lazy, random-access path: ``p.windowed(ds).transform(p)``
        previews the preprocessor, on any view of *ds*, a split included (its
        windows are then over the split's frames). ``run_preprocess`` builds
        the same windows from a sliding queue, reading each frame once.
        """
        if self.window_size == 1:
            return dataset
        from apairo.core.window_view import WindowView

        # WindowView directly, not dataset.window(): the runner also windows
        # its own row-zipping view, which is not an AbstractDataset.
        return WindowView(
            dataset,
            self.window_size,
            self.window_stride,
            reduce=_AnchorStamped(self.window_reduce),
            boundary="clip",
        )

    @abstractmethod
    def __call__(self, sample: Sample) -> Any:
        """Process one frame.

        Args:
            sample: A :class:`~apairo.core.sample.Sample` whose ``data`` dict
                contains at least the keys declared in :attr:`input_keys`.

        Returns:
            A ``numpy.ndarray`` representing the output for this frame.
        """
        ...


class SequencePreprocessor(Preprocessor):
    """Preprocessor that operates on the full sequence at once.

    The runner calls the instance with an iterator over all input frames.
    Use this for algorithms that need global context (ICP, trajectory
    smoothing, …).  Global context is also why a sequence preprocessor
    cannot run lazily: it must be materialized via ``run_preprocess``.

    Output is stored as a single ``{output_key}.npy`` file when
    ``output_loader`` is ``"npy"``.

    Example::

        class GICPPoses(SequencePreprocessor):
            output_key    = "gicp_poses"
            output_loader = "npy"
            input_keys    = ["velodyne_0"]
            sources       = ["velodyne_0"]   # has its own timestamps.txt

            def __call__(self, frames: Iterator[Sample]) -> np.ndarray:
                poses = []
                for sample in frames:
                    pts = sample.data["velodyne_0"]
                    poses.append(register(pts))
                return np.stack(poses)           # (N, 4, 4)
    """

    @abstractmethod
    def __call__(self, frames: Iterator[Sample]) -> Any:
        """Process all frames.

        Args:
            frames: Iterator of :class:`~apairo.core.sample.Sample` objects.

        Returns:
            A ``numpy.ndarray`` of shape ``(N, ...)``.
        """
        ...


class _AnchorStamped:
    """A window reducer whose result must keep the anchor's timestamp: the
    output of a windowed preprocessor belongs to the anchor frame. Module-level
    for picklability."""

    __slots__ = ("reduce",)

    def __init__(self, reduce: Callable[[list[Sample]], Sample]) -> None:
        self.reduce = reduce

    def __call__(self, samples: list[Sample]) -> Sample:
        result = self.reduce(samples)
        if not isinstance(result, Sample):
            raise TypeError(
                f"window_reduce must return a Sample, got {type(result).__name__}."
            )
        if result.timestamp != samples[-1].timestamp:
            raise ValueError(
                f"window_reduce returned timestamp {result.timestamp!r}, the "
                f"anchor frame's is {samples[-1].timestamp!r}: the reduced sample "
                f"must keep the anchor's (samples[-1]) timestamp."
            )
        return result
