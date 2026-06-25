from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from apairo.core.abstract_dataset import AbstractDataset

if TYPE_CHECKING:
    from apairo.core.sample import Sample


class WindowView(AbstractDataset):
    """A view that groups each frame with its temporal neighbours, then reduces.

    Created by :meth:`~apairo.core.abstract_dataset.AbstractDataset.window`.
    Output frame ``j`` is the reduction of a causal window ending at an *anchor*
    frame ``i`` of the parent: the anchor plus the ``size - 1`` preceding frames
    spaced ``stride`` apart, ordered oldest -> newest (``samples[-1]`` is the
    anchor). The membership is pure index arithmetic computed at construction --
    no data is read until access, where the ``reduce`` callable collapses the
    window's samples back into a single :class:`~apairo.core.sample.Sample`.

    Windows never cross a sequence boundary: members are kept only while they
    share the anchor's sequence id (``parent.frame_sequence_ids``). When the
    parent exposes no sequence ids (e.g. a :class:`SynchronizedView` over a
    single async sequence) the whole dataset is treated as one sequence.

    The ``boundary`` policy controls the edges of each sequence:

    * ``"clip"`` -- near a sequence start the window simply holds fewer frames;
      every parent frame yields one output (``len == len(parent)``).
    * ``"drop"`` -- only anchors whose full ``size``-frame window fits within the
      sequence are kept (``len <= len(parent)``).

    Args:
        parent: The dataset to window over (ordered frames; usually synchronous).
        size: Number of frames per window, including the anchor (``>= 1``).
        stride: Gap, in parent frames, between two window members
            (``stride=1`` = consecutive, ``stride=3`` = every third).
        reduce: Callable ``list[Sample] -> Sample`` collapsing a window into one
            sample. **Required** -- a windowed frame holds several samples and is
            not a valid sample until reduced. See ``StackByPose`` in
            ``apairo_transform`` for the point-cloud accumulation policy.
        boundary: ``"clip"`` (default) or ``"drop"`` -- see above.

    Example::

        from apairo_transform import StackByPose

        ds = (apairo.SemanticKittiDataset(root, keys=["lidar", "pose"])
                .window(size=5, stride=1, reduce=StackByPose(time_channel=True)))
        ds[i].data["lidar"]   # densified cloud, correct after .split()/shuffle
    """

    def __init__(
        self,
        parent: AbstractDataset,
        size: int,
        stride: int = 1,
        reduce: Callable[[list["Sample"]], "Sample"] | None = None,
        boundary: str = "clip",
    ) -> None:
        if not isinstance(size, int) or size < 1:
            raise ValueError(f"size must be a positive integer, got {size!r}")
        if not isinstance(stride, int) or stride < 1:
            raise ValueError(f"stride must be a positive integer, got {stride!r}")
        if boundary not in ("clip", "drop"):
            raise ValueError(f"boundary must be 'clip' or 'drop', got {boundary!r}")
        if reduce is None or not callable(reduce):
            raise TypeError(
                "window() requires a reduce callable (list[Sample] -> Sample): a "
                "windowed frame holds several samples and must be reduced back to "
                "one. Pass e.g. reduce=StackByPose() from apairo_transform."
            )

        self._parent = parent
        self._size = size
        self._stride = stride
        self._reduce = reduce
        self._boundary = boundary

        seq = getattr(parent, "frame_sequence_ids", None)  # None -> one sequence
        windows: list[np.ndarray] = []
        anchors: list[int] = []
        for i in range(len(parent)):
            members = [i - k * stride for k in range(size) if i - k * stride >= 0]
            if seq is not None:
                members = [m for m in members if seq[m] == seq[i]]
            members = members[::-1]  # oldest -> newest, anchor (i) last
            if boundary == "drop" and len(members) < size:
                continue
            windows.append(np.asarray(members, dtype=np.int64))
            anchors.append(i)

        self._windows = windows
        self._anchors = np.asarray(anchors, dtype=np.int64)
        self._seq = seq  # resolved once; None means single-sequence parent

    @property
    def anchors(self) -> np.ndarray:
        """Parent index of the anchor (newest) frame of every output window."""
        return self._anchors

    @property
    def is_synchronous(self) -> bool:
        return self._parent.is_synchronous

    @property
    def frame_sequence_ids(self) -> np.ndarray:
        """Sequence id of each output frame -- the anchor's.

        Served from the array resolved at construction (consistent with the
        single-sequence fallback). Raises :class:`AttributeError` when the
        windowed parent exposes no sequence ids, so ``getattr``-based callers
        keep working.
        """
        if self._seq is None:
            raise AttributeError(
                "frame_sequence_ids is unavailable: the windowed parent exposes none."
            )
        return self._seq[self._anchors]

    @property
    def frame_stems(self) -> np.ndarray:
        """Filename stem of each output frame -- the anchor's (delegated)."""
        return self._parent.frame_stems[self._anchors]

    def __len__(self) -> int:
        return len(self._anchors)

    def _load(self, idx: int) -> "Sample":
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        samples = [self._parent[int(m)] for m in self._windows[idx]]
        return self._reduce(samples)

    def __repr__(self) -> str:
        return (
            f"WindowView(n={len(self)}, size={self._size}, stride={self._stride}, "
            f"boundary={self._boundary!r}, parent={self._parent.__class__.__name__})"
        )
