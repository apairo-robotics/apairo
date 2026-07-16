"""Tests for filter_sequences, filter_split, frame_sequence_ids, frame_stems."""

import numpy as np
import pytest

from apairo.core.abstract_dataset import AbstractDataset
from apairo.core.filtered_view import FilteredView
from apairo.core.sample import Sample


class _SeqDataset(AbstractDataset):
    """Minimal dataset with sequence metadata for testing."""

    def __init__(self, seq_ids, stems=None, channel_ids=None):
        # seq_ids: list of sequence ID per frame (e.g. ["seq0", "seq0", "seq1"])
        self._seq_ids = list(seq_ids)
        self._stems = stems or [f"{i:06d}" for i in range(len(seq_ids))]
        self._channel_ids = channel_ids or ["data"] * len(seq_ids)
        self._keys = ["data"]

    def __len__(self):
        return len(self._seq_ids)

    def _load(self, idx):
        return Sample(data={"data": np.array([idx])})

    @property
    def frame_sequence_ids(self):
        return np.array(self._seq_ids, dtype=object)

    @property
    def frame_stems(self):
        return np.array(self._stems, dtype=object)

    @property
    def frame_channel_ids(self):
        return np.array(self._channel_ids, dtype=object)


@pytest.fixture
def ds():
    # 6 frames across 3 sequences
    return _SeqDataset(["seq0", "seq0", "seq1", "seq1", "seq2", "seq2"])


# ---------------------------------------------------------------- filter_sequences


def test_filter_sequences_keeps_matching_frames(ds):
    view = ds.filter_sequences(["seq0", "seq1"])
    assert len(view) == 4
    assert isinstance(view, FilteredView)


def test_filter_sequences_correct_indices(ds):
    view = ds.filter_sequences(["seq2"])
    np.testing.assert_array_equal(view.indices, [4, 5])


def test_filter_sequences_empty_result(ds):
    view = ds.filter_sequences(["nonexistent"])
    assert len(view) == 0


def test_filter_sequences_on_filtered_view(ds):
    # Pre-filter to frames 1..5, then sequence-filter
    inner = ds.filter([1, 2, 3, 4, 5])
    view = inner.filter_sequences(["seq1"])
    # seq1 frames are global 2,3 — local indices 1,2 in inner
    assert len(view) == 2
    assert set(view[0].data["data"].tolist() + view[1].data["data"].tolist()) == {2, 3}


def test_filter_sequences_on_channel_view(ds):
    cv = ds.select(["data"])
    view = cv.filter_sequences(["seq0"])
    assert len(view) == 2


def test_filter_sequences_no_frame_sequence_ids_raises():
    class _Plain(AbstractDataset):
        def __len__(self):
            return 2

        def _load(self, idx):
            return Sample(data={"x": np.array([idx])})

    plain = _Plain()
    plain._keys = ["x"]
    with pytest.raises(AttributeError):
        plain.filter_sequences(["seq0"])


# ---------------------------------------------------------------- frame_sequence_ids delegation


def test_filtered_view_frame_sequence_ids(ds):
    view = ds.filter([0, 2, 4])
    np.testing.assert_array_equal(view.frame_sequence_ids, ["seq0", "seq1", "seq2"])


def test_filtered_view_frame_stems(ds):
    view = ds.filter([0, 2, 4])
    np.testing.assert_array_equal(view.frame_stems, ["000000", "000002", "000004"])


def test_channel_view_delegates_frame_sequence_ids(ds):
    cv = ds.select(["data"])
    np.testing.assert_array_equal(cv.frame_sequence_ids, ds.frame_sequence_ids)


def test_channel_view_delegates_frame_stems(ds):
    cv = ds.select(["data"])
    np.testing.assert_array_equal(cv.frame_stems, ds.frame_stems)


def test_double_filtered_view_frame_sequence_ids(ds):
    inner = ds.filter([0, 1, 2, 3])  # seq0, seq0, seq1, seq1
    outer = inner.filter([1, 3])  # local 1→global 1 (seq0), local 3→global 3 (seq1)
    np.testing.assert_array_equal(outer.frame_sequence_ids, ["seq0", "seq1"])


def test_filtered_view_frame_channel_ids(ds):
    view = ds.filter([0, 2, 4])
    np.testing.assert_array_equal(
        view.frame_channel_ids, ds.frame_channel_ids[[0, 2, 4]]
    )


def test_channel_view_delegates_frame_channel_ids(ds):
    cv = ds.select(["data"])
    np.testing.assert_array_equal(cv.frame_channel_ids, ds.frame_channel_ids)


# ---------------------------------------------------------------- CachedDataset.cache() warning


def test_cache_on_cached_logs_warning(ds, caplog):
    import logging

    cached = ds.cache()
    with caplog.at_level(logging.WARNING, logger="apairo.core.cached_dataset"):
        cached2 = cached.cache()
    assert "already-cached" in caplog.text
    assert len(cached2) == len(ds)
