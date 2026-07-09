"""Calibration.get_tf -- frame-graph resolution, standalone and via ds.calibration."""

import numpy as np
import pytest

from apairo import RawDataset
from apairo.core.config import (
    Calibration,
    read_calibration,
    register_intrinsics,
    register_static_transform,
)


def _t(x, y, z):
    T = np.eye(4)
    T[:3, 3] = (x, y, z)
    return T


# ── Calibration.get_tf ───────────────────────────────────────────────────────


def test_get_tf_multi_hop_and_inverse():
    cal = Calibration({"a_to_b": _t(1, 0, 0), "b_to_c": _t(0, 2, 0)})
    # edges are T_parent_from_child; get_tf(source, target) = T_target_from_source
    np.testing.assert_allclose(cal.get_tf("a", "c")[:3, 3], [-1, -2, 0])
    np.testing.assert_allclose(cal.get_tf("c", "a")[:3, 3], [1, 2, 0])
    np.testing.assert_allclose(cal.get_tf("a", "a"), np.eye(4))


def test_get_tf_is_round_trip():
    cal = Calibration({"a_to_b": _t(1, 0, 0), "b_to_c": _t(0, 2, 0)})
    np.testing.assert_allclose(
        cal.get_tf("a", "c") @ cal.get_tf("c", "a"), np.eye(4), atol=1e-12
    )


def test_calibration_is_a_dict():
    cal = Calibration({"a_to_b": _t(1, 0, 0)})
    assert cal["a_to_b"][0, 3] == 1.0
    assert list(cal) == ["a_to_b"]


def test_get_tf_branch_through_common_parent():
    # base->b and base->c; c to b crosses the shared parent: inv(Tb) @ Tc.
    cal = Calibration({"base_to_b": _t(1, 0, 0), "base_to_c": _t(0, 1, 0)})
    np.testing.assert_allclose(cal.get_tf("c", "b")[:3, 3], [-1, 1, 0])


def test_get_tf_unreachable_raises():
    with pytest.raises(KeyError, match="No static-transform path"):
        Calibration({"a_to_b": _t(1, 0, 0)}).get_tf("a", "z")


def test_get_tf_malformed_key_raises():
    with pytest.raises(ValueError, match="<parent>_to_<child>"):
        Calibration({"a__b": _t(1, 0, 0)}).get_tf("a", "b")


# ── ds.calibration.get_tf, end to end ────────────────────────────────────────


def _mini_seq(tmp_path):
    d = tmp_path / "lidar"
    d.mkdir()
    for i in range(3):
        np.save(d / f"{i:06d}.npy", np.zeros((4, 3), dtype=np.float32))
    np.savetxt(d / "timestamps.txt", np.arange(3))
    RawDataset.init(tmp_path)
    return tmp_path


def test_dataset_calibration_resolves(tmp_path):
    root = _mini_seq(tmp_path)
    register_static_transform(root, "os_lidar", "base_link", _t(0, 0, 1))
    ds = RawDataset(root, keys=["lidar"])
    np.testing.assert_allclose(
        ds.calibration.get_tf("os_lidar", "base_link")[:3, 3], [0, 0, -1]
    )


# ── camera intrinsics ─────────────────────────────────────────────────────────


def test_register_intrinsics_roundtrip(tmp_path):
    K = [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
    register_intrinsics(
        tmp_path,
        "multisense_left",
        K=K,
        distortion=[0.1, -0.05, 0.0, 0.0, 0.0],
        width=640,
        height=480,
    )
    cam = read_calibration(tmp_path).get_intrinsics("multisense_left")
    assert cam.K.shape == (3, 3) and cam.K.dtype == np.float64
    np.testing.assert_allclose(cam.K, K)
    np.testing.assert_allclose(cam.distortion, [0.1, -0.05, 0.0, 0.0, 0.0])
    assert cam.model == "plumb_bob"
    assert (cam.width, cam.height) == (640, 480)
    assert cam.R is None and cam.P is None


def test_intrinsics_rectified_camera_defaults(tmp_path):
    # A rectified image: no distortion recorded, optional R/P present.
    register_intrinsics(tmp_path, "cam", K=np.eye(3), R=np.eye(3), P=np.eye(3, 4))
    cam = read_calibration(tmp_path).get_intrinsics("cam")
    assert cam.distortion.shape == (0,)
    assert cam.R.shape == (3, 3) and cam.P.shape == (3, 4)


def test_get_intrinsics_unknown_camera_lists_available(tmp_path):
    register_intrinsics(tmp_path, "cam_a", K=np.eye(3))
    with pytest.raises(KeyError, match="cam_a"):
        read_calibration(tmp_path).get_intrinsics("cam_b")


def test_transforms_and_cameras_coexist(tmp_path):
    """Registering one kind must not drop the other -- regression: the rewrite
    used to emit only {version, transforms}."""
    register_intrinsics(tmp_path, "cam", K=np.eye(3))
    register_static_transform(tmp_path, "base", "cam", _t(1, 0, 0))
    register_intrinsics(tmp_path, "cam2", K=np.eye(3))
    cal = read_calibration(tmp_path)
    assert "base_to_cam" in cal
    assert set(cal.cameras) == {"cam", "cam2"}


def test_dataset_exposes_intrinsics(tmp_path):
    root = _mini_seq(tmp_path)
    register_intrinsics(root, "cam_left", K=np.diag([500.0, 500.0, 1.0]))
    ds = RawDataset(root, keys=["lidar"])
    assert ds.calibration.get_intrinsics("cam_left").K[0, 0] == 500.0


def test_root_merges_intrinsics_across_sequences(tmp_path):
    for seq, cam in [("seq_a", "cam_a"), ("seq_b", "cam_b")]:
        d = tmp_path / seq / "lidar"
        d.mkdir(parents=True)
        for i in range(2):
            np.save(d / f"{i:06d}.npy", np.zeros((4, 3), dtype=np.float32))
        np.savetxt(d / "timestamps.txt", np.arange(2))
        register_intrinsics(tmp_path / seq, cam, K=np.eye(3))
    ds = RawDataset(tmp_path)
    assert set(ds.calibration.cameras) == {"cam_a", "cam_b"}
