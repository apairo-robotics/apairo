"""resample_pose_path: clamping, circular yaw, axis-agnostic resampling."""

import numpy as np

from apairo.utils import cumulative_distance, resample_pose_path


def _straight_path(n=11):
    """1 m/s along +x, one sample per second."""
    ts = np.arange(n, dtype=np.float64)
    pos = np.stack([ts, np.zeros(n)], axis=1)
    yaw = np.zeros(n)
    return ts, pos, yaw


def test_temporal_interpolation_exact():
    ts, pos, yaw = _straight_path()
    out_pos, out_yaw, _ = resample_pose_path(ts, pos, yaw, np.array([2.5, 7.25]))
    np.testing.assert_allclose(out_pos[:, 0], [2.5, 7.25])
    np.testing.assert_allclose(out_pos[:, 1], 0.0)
    np.testing.assert_allclose(out_yaw, 0.0)


def test_out_of_range_targets_clamp():
    ts, pos, yaw = _straight_path()
    out_pos, _, _ = resample_pose_path(ts, pos, yaw, np.array([-5.0, 100.0]))
    np.testing.assert_allclose(out_pos[:, 0], [0.0, 10.0])


def test_circular_yaw_shortest_path():
    # Heading crosses the ±pi seam: 3.1 -> -3.1 must pass through pi,
    # not through 0.
    ts = np.array([0.0, 1.0])
    pos = np.zeros((2, 2))
    yaw = np.array([3.1, -3.1])
    _, out_yaw, _ = resample_pose_path(ts, pos, yaw, np.array([0.5]))
    assert abs(abs(out_yaw[0]) - np.pi) < 1e-9


def test_spatial_axis_with_timestamps():
    ts, pos, yaw = _straight_path()
    dists = cumulative_distance(pos)
    np.testing.assert_allclose(dists, np.arange(11.0))

    out_pos, _, out_ts = resample_pose_path(
        dists, pos, yaw, np.array([0.0, 3.5]), timestamps=ts
    )
    np.testing.assert_allclose(out_pos[:, 0], [0.0, 3.5])
    np.testing.assert_allclose(out_ts, [0.0, 3.5])   # 1 m/s -> t == d


def test_empty_path():
    out_pos, out_yaw, out_ts = resample_pose_path(
        np.array([]), np.zeros((0, 2)), np.array([]), np.array([1.0, 2.0]),
        timestamps=np.array([]),
    )
    assert out_pos.shape == (2, 2) and (out_pos == 0).all()
    assert (out_yaw == 0).all() and (out_ts == 0).all()


def test_single_point_path():
    out_pos, out_yaw, _ = resample_pose_path(
        np.array([0.0]), np.array([[4.0, 2.0]]), np.array([0.7]),
        np.array([0.0, 9.0]),
    )
    np.testing.assert_allclose(out_pos, [[4.0, 2.0], [4.0, 2.0]])
    np.testing.assert_allclose(out_yaw, 0.7)
