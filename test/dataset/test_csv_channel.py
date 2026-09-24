"""A ``csv`` channel end to end: the layout names the table and its clock
column, the dataset reads the rows as frames and aligns them like any other
channel.

Two real-world layouts, reduced to a few frames: EuRoC MAV (camera images named
by their nanosecond stamp, IMU in ``imu0/data.csv``) and TUM RGB-D (images named
``<sec>.<usec>.png``, the trajectory in ``groundtruth.txt`` at the sequence root).
"""

import numpy as np
import pytest

from apairo.core.config import verify_config, write_config
from apairo.dataset.async_layout.dataset import _detect_loader
from apairo.dataset.raw import RawDataset

T0 = 1403636579758555392  # a EuRoC MH_01 epoch, in ns


@pytest.fixture
def euroc(tmp_path):
    cam = tmp_path / "cam0" / "data"
    cam.mkdir(parents=True)
    for i in range(4):  # 20 Hz
        np.save(cam / f"{T0 + i * 50_000_000}.npy", np.full((2, 2), i, np.float32))
    imu = tmp_path / "imu0"
    imu.mkdir()
    lines = ["#timestamp [ns],w_x [rad s^-1],a_z [m s^-2]"]
    lines += [f"{T0 + k * 5_000_000},{k},{9.81}" for k in range(40)]  # 200 Hz
    (imu / "data.csv").write_text("\n".join(lines) + "\n")
    write_config(
        tmp_path,
        {
            "version": 1,
            "channels": {
                "cam0": {
                    "kind": "raw",
                    "loader": "npys",
                    "directory": "cam0/data",
                    "key": {"name": r"(\d+)", "units": ["ns"]},
                },
                "imu0": {
                    "kind": "raw",
                    "loader": "csv",
                    "key": {"column": "timestamp", "units": ["ns"]},
                },
            },
        },
    )
    return tmp_path


def test_rows_are_frames_on_their_own_clock(euroc):
    ds = RawDataset(euroc, keys=["imu0"])
    assert len(ds) == 40
    assert ds.shape == {"imu0": (2,)}
    np.testing.assert_allclose(ds[3].data["imu0"], [3.0, 9.81])
    assert ds[1].timestamp - ds[0].timestamp == pytest.approx(0.005, abs=1e-6)


def test_imu_rows_align_onto_the_camera_clock(euroc):
    ds = RawDataset(euroc, keys=["cam0", "imu0"])
    view = ds.synchronize(reference="cam0", method="nearest", tolerance=0.003)
    assert len(view) == 4
    for i in range(4):  # camera frame i sits on IMU row 10 * i
        np.testing.assert_allclose(view[i].data["imu0"], [10.0 * i, 9.81])
    assert np.all(np.abs(view.time_offsets("imu0")) < 1e-6)


def test_verify_accepts_the_layout(euroc):
    assert verify_config(euroc) == []


def test_detects_a_csv_directory(euroc):
    assert _detect_loader(euroc / "imu0") == "csv"


def test_tum_table_at_the_sequence_root(tmp_path):
    rgb = tmp_path / "rgb"
    rgb.mkdir()
    for i, t in enumerate(["1305031102.175304", "1305031102.211214"]):
        np.save(rgb / f"{t}.npy", np.full((2, 2), i, np.float32))
    (tmp_path / "groundtruth.txt").write_text(
        "# ground truth trajectory\n"
        "# timestamp tx ty tz qx qy qz qw\n"
        "1305031102.1700 1 0 0 0 0 0 1\n"
        "1305031102.2100 2 0 0 0 0 0 1\n"
    )
    write_config(
        tmp_path,
        {
            "version": 1,
            "channels": {
                "rgb": {
                    "kind": "raw",
                    "loader": "npys",
                    "key": {"name": r"(\d+)\.(\d+)"},
                },
                "groundtruth": {
                    "kind": "raw",
                    "loader": "csv",
                    "directory": ".",
                    "array_file": "groundtruth.txt",
                    "key": {"column": "timestamp"},
                },
            },
        },
    )
    assert verify_config(tmp_path) == []
    ds = RawDataset(tmp_path, keys=["rgb", "groundtruth"])
    view = ds.synchronize(reference="rgb", method="nearest", tolerance=0.01)
    np.testing.assert_allclose(view[0].data["groundtruth"][0], 1.0)
    np.testing.assert_allclose(view[1].data["groundtruth"][0], 2.0)


@pytest.mark.parametrize(
    "meta, match",
    [
        ({"loader": "csv", "key": {"name": r"(\d+)"}}, "no per-frame filenames"),
        ({"loader": "npys", "key": {"column": 0}}, "needs the 'csv' loader"),
        ({"loader": "csv", "key": {"column": -1}}, "column index"),
        ({"loader": "csv", "key": {"column": 0, "units": ["ns", "s"]}}, "one-entry"),
        ({"loader": "csv", "key": {"column": 0, "file": "t.txt"}}, "exactly one of"),
    ],
)
def test_verify_rejects_malformed_column_keys(euroc, meta, match):
    write_config(euroc, {"version": 1, "channels": {"imu0": {"kind": "raw", **meta}}})
    assert any(match in issue for issue in verify_config(euroc))


def test_status_reads_rows_rate_and_width_off_the_table(euroc, capsys):
    import json

    from apairo.cli import main

    try:
        main(["status", str(euroc), "--json"])
    except SystemExit as exc:
        assert exc.code in (0, None)
    data = json.loads(capsys.readouterr().out)
    imu = data["channels"]["imu0"]
    assert imu["loader"] == "csv"
    assert imu["frames"] == 40
    assert imu["rate_hz"] == pytest.approx(200.0, rel=1e-3)
    assert imu["shape"] == [2]
    assert data["events"] == 44
