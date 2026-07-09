"""Calibration.get_tf -- frame-graph resolution, standalone and via ds.calibration."""

import numpy as np
import pytest

from apairo import RawDataset
from apairo.core.config import Calibration, register_static_transform


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
