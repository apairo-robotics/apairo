"""A ``pcd`` channel end to end: channels.yaml declares the field contract, the
dataset resolves it, and ``ds[i].data[key]`` is the projected ``(N, C)`` array.

This is the seam the loader unit tests do not cover: that ``fields:`` travels
from the layout to the loader, and that auto-detection recognizes a PCD
directory at all.
"""

import numpy as np
import pytest

from apairo.core.config import register_raw_channel, verify_config, write_config
from apairo.dataset.async_layout.dataset import _detect_loader
from apairo.dataset.raw import RawDataset
from test.loader.test_pcd_loader import XYZI, write_binary


def _make_pcd_channel(seq_dir, name, clouds, ts, *, writer=write_binary, fields=XYZI):
    d = seq_dir / name
    d.mkdir(parents=True)
    for i, rows in enumerate(clouds):
        writer(d / f"{i:06d}.pcd", fields, rows)
    np.savetxt(d / "timestamps.txt", np.asarray(ts, dtype=float))
    return d


CLOUDS = [
    [[1.0, 2.0, 3.0, 10.0], [4.0, 5.0, 6.0, 20.0]],
    [[7.0, 8.0, 9.0, 30.0]],
]


@pytest.fixture
def seq(tmp_path):
    _make_pcd_channel(tmp_path, "ouster_points", CLOUDS, [0.0, 0.1])
    write_config(
        tmp_path,
        {
            "version": 1,
            "channels": {
                "ouster_points": {
                    "kind": "raw",
                    "loader": "pcd",
                    "frame": "os_sensor",
                    "fields": ["x", "y", "z", "intensity"],
                }
            },
        },
    )
    return tmp_path


def test_loads_the_declared_projection(seq):
    ds = RawDataset(seq, keys=["ouster_points"])
    assert len(ds) == 2
    np.testing.assert_allclose(ds[0].data["ouster_points"], np.array(CLOUDS[0]))
    assert ds[0].data["ouster_points"].shape == (2, 4)
    assert ds[1].data["ouster_points"].shape == (1, 4)


def test_fields_reorder_and_narrow_the_channel(tmp_path):
    _make_pcd_channel(tmp_path, "ouster_points", CLOUDS, [0.0, 0.1])
    register_raw_channel(
        tmp_path, "ouster_points", "pcd", fields=["intensity", "x"], frame="os_sensor"
    )
    ds = RawDataset(tmp_path, keys=["ouster_points"])
    np.testing.assert_allclose(ds[0].data["ouster_points"], [[10.0, 1.0], [20.0, 4.0]])


def test_undeclared_fields_fall_back_to_the_header(tmp_path):
    _make_pcd_channel(tmp_path, "ouster_points", CLOUDS, [0.0, 0.1])
    register_raw_channel(tmp_path, "ouster_points", "pcd")
    ds = RawDataset(tmp_path, keys=["ouster_points"])
    assert ds[0].data["ouster_points"].shape == (2, 4)


def test_a_frame_missing_a_declared_field_names_both_sets(tmp_path):
    """The contract is the point: a vendor frame that drops `intensity` must not
    silently produce a narrower cloud."""
    d = tmp_path / "ouster_points"
    d.mkdir()
    write_binary(d / "000000.pcd", XYZI, CLOUDS[0])
    write_binary(d / "000001.pcd", XYZI[:3], [r[:3] for r in CLOUDS[1]])
    np.savetxt(d / "timestamps.txt", np.array([0.0, 0.1]))
    register_raw_channel(tmp_path, "ouster_points", "pcd", fields=["x", "intensity"])
    ds = RawDataset(tmp_path, keys=["ouster_points"])
    assert ds[0].data["ouster_points"].shape == (2, 2)
    with pytest.raises(ValueError, match="intensity"):
        ds[1]


def test_transforms_apply_to_a_pcd_channel(seq):
    ds = RawDataset(seq, keys=["ouster_points"]).transform(
        "ouster_points", lambda a: a[:, :3]
    )
    assert ds[0].data["ouster_points"].shape == (2, 3)


def test_detection_recognizes_a_pcd_directory(tmp_path):
    _make_pcd_channel(tmp_path, "ouster_points", CLOUDS, [0.0, 0.1])
    assert _detect_loader(tmp_path / "ouster_points") == "pcd"


def test_init_registers_a_pcd_channel(tmp_path):
    _make_pcd_channel(tmp_path, "ouster_points", CLOUDS, [0.0, 0.1])
    RawDataset.init(tmp_path)
    ds = RawDataset(tmp_path)
    assert ds.keys == ["ouster_points"]
    assert ds[0].data["ouster_points"].shape == (2, 4)


# ------------------------------------------------------------------ validation


def test_verify_config_accepts_a_declared_pcd_channel(seq):
    assert verify_config(seq) == []


def test_verify_config_rejects_fields_on_another_loader(tmp_path):
    (tmp_path / "lidar").mkdir()
    write_config(
        tmp_path,
        {
            "version": 1,
            "channels": {
                "lidar": {"kind": "raw", "loader": "bin", "fields": ["x", "y"]}
            },
        },
    )
    assert any(
        "only meaningful for the 'pcd' loader" in i for i in verify_config(tmp_path)
    )


@pytest.mark.parametrize(
    "fields", [[], "x y z", ["x", 3], ["x", "x"]], ids=["empty", "str", "int", "dup"]
)
def test_verify_config_rejects_a_malformed_field_list(tmp_path, fields):
    (tmp_path / "pts").mkdir()
    write_config(
        tmp_path,
        {
            "version": 1,
            "channels": {"pts": {"kind": "raw", "loader": "pcd", "fields": fields}},
        },
    )
    assert any("'fields'" in i for i in verify_config(tmp_path))
