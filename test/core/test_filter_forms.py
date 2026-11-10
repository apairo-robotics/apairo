"""filter(): a boolean mask is a mask, and the per-channel predicate sees the
stored channel on every view."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from apairo.dataset.stream import StreamDataset


@pytest.fixture(autouse=True)
def _no_in_place_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        yield


@pytest.fixture
def ds():
    t = np.arange(4) * 0.1
    values = [np.full(2, float(i)) for i in range(4)]
    return StreamDataset({"a": (t, values)}).synchronize(reference="a")


def test_a_boolean_mask_keeps_the_true_frames(ds):
    view = ds.filter(np.array([True, False, True, False]))
    assert view.indices.tolist() == [0, 2]
    assert ds.filter([False, False, False, True]).indices.tolist() == [3]


def test_a_mask_of_the_wrong_length_is_refused(ds):
    with pytest.raises(ValueError, match="one entry per frame"):
        ds.filter(np.array([True, False]))


def test_indices_still_work_and_are_checked(ds):
    assert ds.filter([1, 0, 1]).indices.tolist() == [1, 0, 1]
    assert len(ds.filter([])) == 0
    with pytest.raises(IndexError, match="index 7 is out of range"):
        ds.filter([0, 7])
    with pytest.raises(TypeError, match="integers.*or a boolean mask"):
        ds.filter(np.array([0.5, 1.5]))


def test_the_per_channel_predicate_sees_stored_data_on_every_view(ds):
    scaled = ds.transform("a", lambda x: x * 10, in_place=False)

    def keep(v):
        return v[0] >= 2  # true for stored 2, 3; for transformed 10, 20, 30

    assert scaled.filter("a", keep).indices.tolist() == [2, 3]
    assert scaled.filter([0, 1, 2, 3]).filter("a", keep).indices.tolist() == [2, 3]
    assert scaled.select(["a"]).filter("a", keep).indices.tolist() == [2, 3]
    assert scaled.filter([3, 2, 1]).filter("a", keep).indices.tolist() == [0, 1]


def test_a_channel_published_by_a_transform_can_be_filtered_on(ds):
    derived = ds.transform("a", lambda x: x * 10, output="b", in_place=False)
    assert derived.filter("b", lambda v: v[0] >= 20).indices.tolist() == [2, 3]
    with pytest.raises(KeyError, match="no channel 'nope'"):
        derived.filter("nope", lambda v: True)
