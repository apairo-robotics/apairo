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
    output_loader : str
        Storage format -- ``"npys"`` (one file per frame), ``"npy"`` (single
        stacked file), or ``"bin"`` (raw binary, one file per frame).
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
    output_loader: ClassVar[str]
    input_keys: ClassVar[list[str]]
    timestamps_from: ClassVar[str | None] = None
    sources: ClassVar[list[str] | None] = None

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
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

    Example::

        class TravLabel(FramePreprocessor):
            output_key    = "trav_label"
            output_loader = "npys"
            input_keys    = ["velodyne_0"]
            timestamps_from = "velodyne_0"   # no own timestamps.txt

            def __call__(self, sample: Sample) -> np.ndarray:
                pts = sample.data["velodyne_0"]
                return my_model(pts)
    """

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
