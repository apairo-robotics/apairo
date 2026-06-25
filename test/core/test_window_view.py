import numpy as np
import pytest

from apairo.core.abstract_dataset import AbstractDataset
from apairo.core.sample import Sample


class FakeDataset(AbstractDataset):
    """Minimal ordered dataset: frame i carries its own global index.

    ``seq_ids=None`` hides ``frame_sequence_ids`` (the property raises, so
    ``getattr(parent, "frame_sequence_ids", None)`` falls back to None) to
    exercise the single-sequence path.
    """

    def __init__(self, n, seq_ids=None):
        self._n = n
        self._seq = np.asarray(seq_ids, dtype=object) if seq_ids is not None else None
        self.loads = 0

    @property
    def is_synchronous(self):
        return True

    @property
    def frame_sequence_ids(self):
        if self._seq is None:
            raise AttributeError("frame_sequence_ids")
        return self._seq

    def __len__(self):
        return self._n

    def _load(self, idx):
        self.loads += 1
        return Sample(data={"idx": idx})


def members_reducer(samples):
    """Reduce a window to the list of parent indices it pulled (oldest->newest)."""
    return Sample(data={"members": [s.data["idx"] for s in samples]})


def windows_of(view):
    return [view[j].data["members"] for j in range(len(view))]


# ── boundary: clip ───────────────────────────────────────────────────────────

def test_clip_keeps_every_frame_and_shrinks_at_start():
    ds = FakeDataset(5, ["a", "a", "a", "b", "b"])
    view = ds.window(size=3, stride=1, reduce=members_reducer)
    assert len(view) == len(ds)
    assert windows_of(view) == [[0], [0, 1], [0, 1, 2], [3], [3, 4]]


def test_window_never_crosses_sequence_boundary():
    ds = FakeDataset(5, ["a", "a", "a", "b", "b"])
    view = ds.window(size=4, stride=1, reduce=members_reducer)
    # anchor 3 (seq b) must not reach back into seq a
    assert view[3].data["members"] == [3]
    assert view[4].data["members"] == [3, 4]


def test_anchor_is_last_member():
    ds = FakeDataset(4, ["a", "a", "a", "a"])
    view = ds.window(size=3, stride=1, reduce=members_reducer)
    for j in range(len(view)):
        assert view[j].data["members"][-1] == view.anchors[j]


# ── boundary: drop ───────────────────────────────────────────────────────────

def test_drop_keeps_only_full_windows():
    ds = FakeDataset(5, ["a", "a", "a", "b", "b"])
    view = ds.window(size=3, stride=1, reduce=members_reducer, boundary="drop")
    # seq a yields one full window (anchor 2); seq b is too short
    assert len(view) == 1
    assert view.anchors.tolist() == [2]
    assert view[0].data["members"] == [0, 1, 2]


# ── stride ───────────────────────────────────────────────────────────────────

def test_stride_skips_frames():
    ds = FakeDataset(5, ["a"] * 5)
    view = ds.window(size=2, stride=2, reduce=members_reducer)
    assert view[4].data["members"] == [2, 4]
    assert view[0].data["members"] == [0]


# ── single-sequence fallback ─────────────────────────────────────────────────

def test_fallback_single_sequence_when_no_seq_ids():
    ds = FakeDataset(4)  # no frame_sequence_ids
    view = ds.window(size=2, stride=1, reduce=members_reducer)
    assert windows_of(view) == [[0], [0, 1], [1, 2], [2, 3]]


# ── laziness ─────────────────────────────────────────────────────────────────

def test_construction_reads_no_data():
    ds = FakeDataset(10, ["a"] * 10)
    view = ds.window(size=3, stride=1, reduce=members_reducer)
    assert ds.loads == 0  # only index arithmetic at construction
    _ = view[5]
    assert ds.loads == 3  # one window read on access


# ── chaining ─────────────────────────────────────────────────────────────────

def test_output_frame_sequence_ids_are_anchor_ids():
    ds = FakeDataset(5, ["a", "a", "a", "b", "b"])
    view = ds.window(size=3, stride=1, reduce=members_reducer)
    np.testing.assert_array_equal(
        view.frame_sequence_ids, np.asarray(["a", "a", "a", "b", "b"], dtype=object)
    )


def test_frame_sequence_ids_unavailable_on_single_sequence_parent():
    ds = FakeDataset(4)  # no frame_sequence_ids
    view = ds.window(size=2, stride=1, reduce=members_reducer)
    with pytest.raises(AttributeError):
        _ = view.frame_sequence_ids
    assert getattr(view, "frame_sequence_ids", None) is None  # getattr-friendly


def test_window_then_filter_chains():
    ds = FakeDataset(5, ["a"] * 5)
    view = ds.window(size=2, stride=1, reduce=members_reducer).filter([0, 2, 4])
    assert len(view) == 3
    assert [s.data["members"] for s in view] == [[0], [1, 2], [3, 4]]


def test_downstream_transform_applies():
    ds = FakeDataset(3, ["a"] * 3)
    view = ds.window(size=2, stride=1, reduce=members_reducer)
    view = view.transform(lambda s: Sample(data={"count": len(s.data["members"])}))
    assert [s.data["count"] for s in view] == [1, 2, 2]


# ── validation ───────────────────────────────────────────────────────────────

def test_reduce_is_required():
    ds = FakeDataset(3, ["a"] * 3)
    with pytest.raises(TypeError, match="reduce"):
        ds.window(size=2, stride=1)


@pytest.mark.parametrize(
    "kwargs", [{"size": 0}, {"size": 2, "stride": 0}, {"size": 2, "boundary": "x"}]
)
def test_invalid_params(kwargs):
    ds = FakeDataset(3, ["a"] * 3)
    with pytest.raises((ValueError, TypeError)):
        ds.window(reduce=members_reducer, **kwargs)


def test_out_of_range_access():
    ds = FakeDataset(3, ["a"] * 3)
    view = ds.window(size=2, stride=1, reduce=members_reducer)
    with pytest.raises(IndexError):
        view[len(view)]
