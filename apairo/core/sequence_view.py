from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apairo.core.sample import Sample


class SequenceView:
    """A lightweight view over a contiguous slice of a dataset's frames.

    Wraps any dataset that supports ``__getitem__(int)`` and exposes a
    local index space ``[0, len(view))``.  Both synchronous
    (:class:`~apairo.core.profiled_dataset.ProfiledDataset`) and asynchronous
    (:class:`~apairo.dataset.tartan_kitti.TartanKittiDataset`) datasets expose
    these views via their ``sequence()`` method.

    Args:
        parent: The underlying dataset.
        indices: Global indices in *parent* that belong to this sequence.
        sequence_id: Human-readable name for this sequence (usually the
            directory name, e.g. ``"00000"`` or ``"2024-01-01_forest"``).

    Example::

        seq = ds.sequence("00000")
        len(seq)                # frames in this sequence
        seq[0]                  # first frame (local index)
        for sample in seq:      # iterate
            visualize(sample)
        ds["00000", 3]          # shorthand for seq[3]
    """

    def __init__(self, parent, indices: list[int] | range, sequence_id: str) -> None:
        self._parent = parent
        self._indices = list(indices)
        self._sequence_id = sequence_id

    @property
    def sequence_id(self) -> str:
        return self._sequence_id

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Sample:
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        return self._parent[self._indices[idx]]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        return f"SequenceView(id={self._sequence_id!r}, n_frames={len(self)})"
