"""Tests for SynchronizedView -- async -> sync resampling."""

import numpy as np
import pytest

from apairo import KittiDataset, SynchronizedView
from apairo.core import AbstractDataset, FilteredView, Sample


def _make_async_dataset(tmp_path, lidar_ts, imu_ts):
    """Build a two-channel KittiDataset; frame i of channel c holds np.full(2, i)."""
    for name, ts in [("lidar", lidar_ts), ("imu", imu_ts)]:
        d = tmp_path / name
        d.mkdir()
        for i in range(len(ts)):
            np.save(d / f"{i:06d}.npy", np.full(2, i, dtype=np.float32))
        np.savetxt(d / "timestamps.txt", np.asarray(ts, dtype=float))
    profile = tmp_path / "profile.yaml"
    profile.write_text("lidar: npys\nimu: npys\n")
    return KittiDataset(tmp_path, keys=["lidar", "imu"], dataset_profile=profile)


@pytest.fixture
def async_ds(tmp_path):
    # lidar: 4 events at [0, 1/3, 2/3, 1]; imu: 8 events at [0, 1/7, ..., 1]
    return _make_async_dataset(
        tmp_path, np.linspace(0, 1, 4), np.linspace(0, 1, 8)
    )


def test_default_reference_is_lowest_frequency(async_ds):
    view = async_ds.synchronize()
    assert view.reference == "lidar"
    assert len(view) == 4


def test_latest_matches_last_event_before_ref(async_ds):
    view = async_ds.synchronize(reference="lidar", method="latest")
    np.testing.assert_array_equal(view.frame_indices["lidar"], [0, 1, 2, 3])
    np.testing.assert_array_equal(view.frame_indices["imu"], [0, 2, 4, 7])


def test_nearest_matches_closest_event(async_ds):
    view = async_ds.synchronize(reference="lidar", method="nearest")
    np.testing.assert_array_equal(view.frame_indices["imu"], [0, 2, 5, 7])


def test_samples_are_complete_and_timestamped(async_ds):
    view = async_ds.synchronize(reference="lidar")
    for i in range(len(view)):
        s = view[i]
        assert set(s.data) == {"lidar", "imu"}
        assert s.timestamp == pytest.approx(i / 3)
        np.testing.assert_array_equal(s.data["lidar"], np.full(2, i))
    np.testing.assert_array_equal(view[1].data["imu"], np.full(2, 2))


def test_view_is_synchronous(async_ds):
    assert not async_ds.is_synchronous
    view = async_ds.synchronize()
    assert view.is_synchronous
    assert view.timestamps is None


def test_tolerance_drops_unmatched_frames(async_ds):
    # latest deltas for imu: [0, 1/21, 2/21, 0] -- 2/21 > 0.05 drops frame 2
    view = async_ds.synchronize(reference="lidar", tolerance=0.05)
    assert len(view) == 3
    np.testing.assert_array_equal(view.frame_indices["lidar"], [0, 1, 3])


def test_latest_drops_frames_before_first_event(tmp_path):
    # imu starts after the first lidar frame -> frame 0 has no latest match
    ds = _make_async_dataset(
        tmp_path, np.linspace(0, 1, 4), np.linspace(0.1, 1, 8)
    )
    view = ds.synchronize(reference="lidar", method="latest")
    assert len(view) == 3
    np.testing.assert_array_equal(view.frame_indices["lidar"], [1, 2, 3])


def test_reference_override(async_ds):
    view = async_ds.synchronize(reference="imu")
    assert view.reference == "imu"
    assert len(view) == 8
    np.testing.assert_array_equal(view.frame_indices["imu"], np.arange(8))


def test_time_offsets(async_ds):
    view = async_ds.synchronize(reference="lidar", method="latest")
    offsets = view.time_offsets("imu")
    assert (offsets <= 0).all()  # latest events are never in the future
    np.testing.assert_allclose(view.time_offsets("lidar"), 0.0)


def test_chaining_filter_transform_cache(async_ds):
    view = (
        async_ds.synchronize()
        .transform("lidar", lambda x: x * 10)
        .filter(lambda s: s.data["lidar"][0] >= 10)
    )
    assert isinstance(view, FilteredView)
    assert len(view) == 3
    cached = view.cache()
    np.testing.assert_array_equal(cached[0].data["lidar"], np.full(2, 10.0))


def test_per_channel_filter_works_on_synchronized_view(async_ds):
    view = async_ds.synchronize().filter("lidar", lambda x: x[0] >= 2)
    assert len(view) == 2


def test_errors(async_ds, tmp_path):
    with pytest.raises(KeyError):
        async_ds.synchronize(reference="nope")
    with pytest.raises(ValueError):
        async_ds.synchronize(method="interpolate")

    class SyncDS(AbstractDataset):
        timestamps = None

        def __init__(self):
            self.keys = ["a"]

        def __len__(self):
            return 1

        def _load(self, idx):
            return Sample(data={"a": np.zeros(1)})

    with pytest.raises(ValueError):
        SyncDS().synchronize()


def test_repr(async_ds):
    view = async_ds.synchronize()
    assert "SynchronizedView" in repr(view)
    assert "lidar" in repr(view)
