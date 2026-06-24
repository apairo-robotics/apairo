"""Tests for SynchronizedView -- async -> sync resampling."""

import numpy as np
import pytest

from apairo import Interpolator
from apairo.core import AbstractDataset, FilteredView, Sample
from apairo.dataset.kitti import AsyncLayoutDataset


class LerpInterp(Interpolator):
    """Linear interpolation used as the reference Interpolator in tests."""

    def __call__(self, t, t0, v0, t1, v1):
        a = (t - t0) / (t1 - t0)
        return (1.0 - a) * v0 + a * v1


class NeverCalled(Interpolator):
    """Fails the test if the view calls it (exact-match bypass checks)."""

    def __call__(self, t, t0, v0, t1, v1):
        raise AssertionError("interpolator must not be called here")


def _make_async_dataset(tmp_path, lidar_ts, imu_ts):
    """Build a two-channel AsyncLayoutDataset; frame i of channel c holds np.full(2, i)."""
    for name, ts in [("lidar", lidar_ts), ("imu", imu_ts)]:
        d = tmp_path / name
        d.mkdir()
        for i in range(len(ts)):
            np.save(d / f"{i:06d}.npy", np.full(2, i, dtype=np.float32))
        np.savetxt(d / "timestamps.txt", np.asarray(ts, dtype=float))
    profile = tmp_path / "profile.yaml"
    profile.write_text("lidar: npys\nimu: npys\n")
    return AsyncLayoutDataset(tmp_path, keys=["lidar", "imu"], dataset_profile=profile)


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


# ------------------------------------------------------------------ external clock


def test_external_clock_fixed_rate(async_ds):
    # 10 Hz clock over [0, 1] -> 11 ticks; every tick has lidar+imu history
    clock = np.arange(0.0, 1.05, 0.1)
    view = async_ds.synchronize(reference=clock)
    assert view.reference is None
    assert len(view) == 11
    np.testing.assert_allclose(view.reference_timestamps, clock)
    assert set(view[5].data) == {"lidar", "imu"}


def test_external_clock_drops_ticks_before_first_event(async_ds):
    # ticks before t=0 have no "latest" event in any channel
    clock = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    view = async_ds.synchronize(reference=clock)
    assert len(view) == 3
    np.testing.assert_allclose(view.reference_timestamps, [0.0, 0.5, 1.0])


def test_external_clock_validation(async_ds):
    with pytest.raises(ValueError, match="ascending"):
        async_ds.synchronize(reference=np.array([1.0, 0.5]))
    with pytest.raises(ValueError, match="1-D"):
        async_ds.synchronize(reference=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="1-D"):
        async_ds.synchronize(reference=np.array([]))


def test_distance_clock(async_ds):
    """One frame every 2 m, robot moving at 4 m/s along x then stopping."""
    from apairo.utils import clock_from_distance

    odom_ts = np.linspace(0.0, 1.0, 21)            # 20 Hz over 1 s
    x = np.minimum(odom_ts, 0.5) * 4.0              # moves 2 m, then static
    positions = np.stack([x, np.zeros_like(x)], axis=1)

    clock = clock_from_distance(odom_ts, positions, step=0.5)
    # 2 m at 0.5 m steps -> ticks at 0.0, 0.5, 1.0, 1.5, 2.0 m
    assert len(clock) == 5
    # static second half generates no ticks
    assert clock.max() <= 0.5

    view = async_ds.synchronize(reference=clock, method="nearest")
    assert len(view) == 5
    assert set(view[0].data) == {"lidar", "imu"}


def test_distance_clock_validation():
    from apairo.utils import clock_from_distance

    with pytest.raises(ValueError, match="same length"):
        clock_from_distance(np.zeros(3), np.zeros((2, 2)), step=1.0)
    with pytest.raises(ValueError, match="positive"):
        clock_from_distance(np.zeros(3), np.zeros((3, 2)), step=0.0)


# ------------------------------------------------------------------ custom method


def test_callable_method(async_ds):
    """A custom strategy: always pick the first event of each channel."""
    def first_event(ts: np.ndarray, ref_ts: np.ndarray) -> np.ndarray:
        return np.zeros(len(ref_ts), dtype=np.int64)

    view = async_ds.synchronize(reference="lidar", method=first_event)
    assert len(view) == 4
    np.testing.assert_array_equal(view.frame_indices["imu"], [0, 0, 0, 0])
    assert "first_event" in repr(view)


def test_callable_method_negative_index_drops_frame(async_ds):
    """Negative indices mark 'no match' and drop the reference frame."""
    def skip_first(ts: np.ndarray, ref_ts: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(ts, ref_ts, side="right") - 1
        idx[0] = -1
        return idx

    view = async_ds.synchronize(reference="lidar", method=skip_first)
    assert len(view) == 3


def test_callable_method_bad_shape_raises(async_ds):
    with pytest.raises(ValueError, match="shape"):
        async_ds.synchronize(method=lambda ts, ref: np.zeros(1, dtype=int))


def test_invalid_method_string(async_ds):
    with pytest.raises(ValueError, match="callable"):
        async_ds.synchronize(method="interpolate")


# ------------------------------------------------------------------ interpolation


def test_interpolated_channel_value(async_ds):
    """imu frame i holds np.full(2, i): interpolation blends event values."""
    view = async_ds.synchronize(
        reference="lidar", method={"imu": LerpInterp()}
    )
    assert len(view) == 4
    # lidar tick t=1/3 sits between imu events 2 (t=2/7) and 3 (t=3/7):
    # alpha = (1/3 - 2/7) / (1/7) = 1/3 -> value 2 + 1/3
    np.testing.assert_allclose(
        view[1].data["imu"], np.full(2, 2 + 1 / 3), rtol=1e-6  # data is float32
    )
    # the lidar channel (unlisted) defaults to "latest" -- raw events
    np.testing.assert_array_equal(view[1].data["lidar"], np.full(2, 1))


def test_interpolation_exact_tick_returns_stored_value(async_ds):
    # lidar ticks 0 and 1 coincide exactly with imu events 0 and 7
    view = async_ds.synchronize(reference="lidar", method={"imu": LerpInterp()})
    np.testing.assert_allclose(view[0].data["imu"], np.full(2, 0.0))
    np.testing.assert_allclose(view[3].data["imu"], np.full(2, 7.0))


def test_interpolation_exact_last_event_bypasses_interpolator(tmp_path):
    # single shared clock: every tick is an exact match -> never interpolate
    ts = np.linspace(0, 1, 4)
    ds = _make_async_dataset(tmp_path, ts, ts)
    view = ds.synchronize(reference="lidar", method={"imu": NeverCalled()})
    assert len(view) == 4
    np.testing.assert_array_equal(view[2].data["imu"], np.full(2, 2))


def test_interpolation_requires_bracketing(tmp_path):
    # imu starts after the first lidar tick and ends before the last one
    ds = _make_async_dataset(
        tmp_path, np.linspace(0, 1, 4), np.linspace(0.2, 0.8, 5)
    )
    view = ds.synchronize(reference="lidar", method={"imu": LerpInterp()})
    # ticks 0.0 and 1.0 are not bracketed by imu events -> dropped
    assert len(view) == 2
    np.testing.assert_allclose(view.reference_timestamps, [1 / 3, 2 / 3])


def test_interpolation_tolerance_applies_to_both_neighbours(async_ds):
    # imu gaps are 1/7 s; ticks 1/3 and 2/3 have a neighbour > 0.06 s away
    view = async_ds.synchronize(
        reference="lidar", method={"imu": LerpInterp()}, tolerance=0.06
    )
    np.testing.assert_allclose(view.reference_timestamps, [0.0, 1.0])


def test_method_set_raises(async_ds):
    with pytest.raises(TypeError, match="not a set"):
        async_ds.synchronize(method={"imu", LerpInterp()})  # set, not {imu: ...}


def test_interpolated_frame_indices_and_offsets(async_ds):
    view = async_ds.synchronize(reference="lidar", method={"imu": LerpInterp()})
    assert view.frame_indices["imu"].shape == (4, 2)   # bracketing pairs
    assert view.frame_indices["lidar"].shape == (4,)   # matched
    np.testing.assert_array_equal(view.time_offsets("imu"), np.zeros(4))


def test_interpolator_as_global_method(async_ds):
    """An Interpolator passed directly applies to every channel."""
    view = async_ds.synchronize(reference="lidar", method=LerpInterp())
    assert len(view) == 4
    np.testing.assert_allclose(view[1].data["lidar"], np.full(2, 1.0))


def test_method_dict_unknown_channel_raises(async_ds):
    with pytest.raises(KeyError, match="unknown channels"):
        async_ds.synchronize(method={"nope": LerpInterp()})


def test_method_dict_repr(async_ds):
    view = async_ds.synchronize(method={"imu": LerpInterp()})
    assert "per-channel" in repr(view)
