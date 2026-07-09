"""Tests for StreamDataset -- in-memory asynchronous event streams."""

import numpy as np
import pytest

from apairo import StreamDataset


@pytest.fixture
def streams():
    return {
        "lidar": (np.array([0.0, 0.4, 0.8]), ["scan0", "scan1", "scan2"]),
        "cmd": (np.array([0.1, 0.3, 0.5, 0.7]), [10, 30, 50, 70]),
    }


def test_timeline_is_time_ordered(streams):
    ds = StreamDataset(streams)
    assert len(ds) == 7
    last = -np.inf
    seen = set()
    for sample in ds:
        assert len(sample.data) == 1  # one event, one channel
        assert sample.timestamp >= last
        last = sample.timestamp
        seen.update(sample.data)
    assert seen == {"lidar", "cmd"}


def test_items_are_passed_through_untouched(streams):
    ds = StreamDataset(streams)
    assert ds[0].data["lidar"] == "scan0"  # arbitrary Python objects
    assert ds[1].data["cmd"] == 10


def test_synchronize(streams):
    ds = StreamDataset(streams)
    view = ds.synchronize(reference="lidar", method="previous")
    assert view.is_synchronous
    assert len(view) == 2  # tick 0.0 has no cmd yet
    s = view[0]
    assert s.data == {"lidar": "scan1", "cmd": 30}  # latest at t=0.4
    assert view[1].data["cmd"] == 70


def test_synchronize_external_clock(streams):
    ds = StreamDataset(streams)
    view = ds.synchronize(reference=np.array([0.35, 0.75]))
    assert len(view) == 2
    assert view[0].data == {"lidar": "scan0", "cmd": 30}


def test_validation_errors():
    with pytest.raises(ValueError, match="at least one"):
        StreamDataset({})
    with pytest.raises(ValueError, match="timestamps for"):
        StreamDataset({"a": (np.array([0.0, 1.0]), ["x"])})
    with pytest.raises(ValueError, match="ascending"):
        StreamDataset({"a": (np.array([1.0, 0.0]), ["x", "y"])})
    with pytest.raises(ValueError, match="1-D"):
        StreamDataset({"a": (np.zeros((2, 2)), ["x", "y"])})


def test_repr(streams):
    r = repr(StreamDataset(streams))
    assert "StreamDataset" in r and "events=7" in r
