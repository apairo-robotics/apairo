from __future__ import annotations
from abc import abstractmethod, ABC
from typing import Any, ClassVar, Iterator, Optional

from apairo.core.sample import Sample


class Preprocessor(ABC):
    """Base class for all apairo preprocessors.

    Subclasses declare their I/O contract as class attributes and implement
    :meth:`process`.  The runner (:func:`apairo.preprocess.run`) reads these
    attributes to decide how to load inputs, how to save outputs, and how to
    register the new channel in ``.apairo``.

    Class attributes
    ----------------
    output_key : str
        Subdirectory name for the output channel (e.g. ``"trav_label"``).
    output_loader : str
        Storage format — ``"npys"`` (one file per frame), ``"npy"`` (single
        stacked file), or ``"bin"`` (raw binary, one file per frame).
    input_keys : list[str]
        Dataset channels needed as input.
    timestamps_from : str or None
        If ``None`` the runner writes a ``timestamps.txt`` from the input
        timestamps.  If set to a channel name the output borrows that channel's
        timestamps and no file is written.
    sources : list[str] or None
        Provenance — channels this output was derived from (stored in
        ``.apairo`` for reference).
    """

    output_key: ClassVar[str]
    output_loader: ClassVar[str]
    input_keys: ClassVar[list[str]]
    timestamps_from: ClassVar[Optional[str]] = None
    sources: ClassVar[Optional[list[str]]] = None


class FramePreprocessor(Preprocessor):
    """Preprocessor that operates frame-by-frame.

    The runner calls :meth:`process` once per input frame.  Use this for
    per-scan operations (label inference, feature extraction, …).

    Output is stored as one file per frame (``000000.npy``, ``000001.npy``,
    …) when ``output_loader`` is ``"npys"`` or ``"bin"``.

    Example::

        class TravLabel(FramePreprocessor):
            output_key    = "trav_label"
            output_loader = "npys"
            input_keys    = ["velodyne_0"]
            timestamps_from = "velodyne_0"   # no own timestamps.txt

            def process(self, sample: Sample) -> np.ndarray:
                pts = sample.data["velodyne_0"]
                return my_model(pts)
    """

    @abstractmethod
    def process(self, sample: Sample) -> Any:
        """Process one frame.

        Args:
            sample: A :class:`~apairo.core.sample.Sample` whose ``data`` dict
                contains exactly the keys declared in :attr:`input_keys`.

        Returns:
            A ``numpy.ndarray`` or ``torch.Tensor`` representing the output for
            this frame.
        """
        ...


class SequencePreprocessor(Preprocessor):
    """Preprocessor that operates on the full sequence at once.

    The runner calls :meth:`process` with an iterator over all input frames.
    Use this for algorithms that need global context (ICP, trajectory
    smoothing, …).

    Output is stored as a single ``{output_key}.npy`` file when
    ``output_loader`` is ``"npy"``.

    Example::

        class GICPPoses(SequencePreprocessor):
            output_key    = "gicp_poses"
            output_loader = "npy"
            input_keys    = ["velodyne_0"]
            sources       = ["velodyne_0"]   # has its own timestamps.txt

            def process(self, frames: Iterator[Sample]) -> np.ndarray:
                poses = []
                for sample in frames:
                    pts = sample.data["velodyne_0"]
                    poses.append(register(pts))
                return np.stack(poses)           # (N, 4, 4)
    """

    @abstractmethod
    def process(self, frames: Iterator[Sample]) -> Any:
        """Process all frames.

        Args:
            frames: Iterator of :class:`~apairo.core.sample.Sample` objects.

        Returns:
            A ``numpy.ndarray`` or ``torch.Tensor`` of shape ``(N, ...)``.
        """
        ...
