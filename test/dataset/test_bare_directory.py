"""A directory that holds its data files itself, with no sub-directory, is a
dataset of its own.

Two real shapes motivated it: a channel directory opened on its own
(``seq/velodyne_0``, frames plus ``timestamps.txt``), and a logger's folder of
CSV tables side by side (a robot arm's F/T, IMU and orientation logs). Before,
``init`` answered "no channels" for both. Now the frames are one channel named
after the directory, each table is one channel named after its stem, and every
entry reads from ``directory: "."``. A directory with a sub-directory keeps its
meaning -- a sequence whose channels are its sub-directories, or a root.
"""

import json
import re

import numpy as np
import pytest

from apairo.cli import main
from apairo.core.config import read_config, write_config
from apairo.dataset.async_layout.dataset import _bare_channel_entries, _detect_loader
from apairo.dataset.raw import RawDataset
from test.loader.test_pcd_loader import XYZI, write_binary

T0_US = 1708857503931759  # a microsecond epoch, as in the KUKA logs


def _cli(args, capsys):
    try:
        code = main(args)
    except SystemExit as exc:
        code = exc.code
    return code, capsys.readouterr().out


def _bare_frames(d, n=5, suffix=None):
    d.mkdir(parents=True)
    for i in range(n):
        np.save(d / f"{i:06d}.npy", np.full((4, 3), i, np.float32))
        if suffix:
            np.save(d / f"{i:06d}_{suffix}.npy", np.full(4, i, np.uint8))
    np.savetxt(d / "timestamps.txt", 10.0 + np.arange(n) * 0.1)
    return d


def _table(path, period_us, n, width):
    cols = ",".join(f"c{j}" for j in range(width))
    rows = [
        f"{T0_US + k * period_us}," + ",".join(str(k + j) for j in range(width))
        for k in range(n)
    ]
    path.write_text(f"t,{cols}\n" + "\n".join(rows) + "\n")


@pytest.fixture
def logger_dir(tmp_path):
    """Three CSV tables side by side, each with its clock in column ``t`` (us)."""
    d = tmp_path / "kuka"
    d.mkdir()
    _table(d / "1-baseline_accel.csv", 3937, 12, 3)
    _table(d / "1-baseline_orientations.csv", 10000, 5, 9)
    _table(d / "1-baseline_wrench.csv", 1437, 30, 6)
    return d


# ─────────────────────────────── a bare frame directory ───────────────────────


def test_a_bare_frame_directory_loads_as_one_channel(tmp_path):
    d = _bare_frames(tmp_path / "velodyne_0", suffix="intensity")
    ds = RawDataset(d)
    assert ds.keys == ["velodyne_0", "velodyne_0_intensity"]
    assert len(ds.loaders["velodyne_0"]) == 5
    np.testing.assert_allclose(ds.timestamps["velodyne_0"], 10.0 + np.arange(5) * 0.1)
    np.testing.assert_array_equal(ds.loaders["velodyne_0_intensity"][2], np.full(4, 2))
    channels = read_config(d)["channels"]
    assert channels["velodyne_0"]["directory"] == "."
    assert channels["velodyne_0_intensity"] == {
        "kind": "raw",
        "loader": "npys",
        "directory": ".",
        "suffix": "intensity",
    }


def test_init_status_and_check_accept_a_bare_frame_directory(tmp_path, capsys):
    d = _bare_frames(tmp_path / "velodyne_0", suffix="intensity")
    code, out = _cli(["init", str(d)], capsys)
    assert code in (0, None), out
    _, out = _cli(["status", str(d), "--json"], capsys)
    status = json.loads(out)
    lidar = status["channels"]["velodyne_0"]
    assert lidar["frames"] == 5
    assert lidar["rate_hz"] == pytest.approx(10.0)
    assert status["channels"]["velodyne_0_intensity"]["shape"] == [4]
    assert status["channels"]["velodyne_0_intensity"]["dtype"] == "uint8"
    assert status["issues"] == []
    code, _ = _cli(["check", str(d)], capsys)
    assert code == 0


def test_a_clockless_bare_directory_gets_its_key_from_declare(tmp_path, capsys):
    """Clouds named by their nanosecond stamp, no timestamps.txt -- the shape of
    an evaluation export: init finds the channel, check says it has no clock,
    declare writes the key."""
    d = tmp_path / "scores"
    d.mkdir()
    for k in range(4):
        stamp = 1785248347168248576 + k * 10**9
        write_binary(d / f"flower_pot_{stamp}.pcd", XYZI, [[0.0, 0.0, 0.0, float(k)]])
    _cli(["init", str(d)], capsys)
    code, out = _cli(["check", str(d), "--json"], capsys)
    assert code == 1
    assert any("'scores': no clock" in i for i in json.loads(out)["issues"])

    _cli(["declare", str(d)], capsys)
    declaration = (d / "apairo.yaml").read_text()
    assert 'directory: "."' in declaration
    assert "    key: {name: '(\\d+)$', units: [ns]}" in declaration
    ds = RawDataset(d)
    assert np.diff(ds.timestamps["scores"]) == pytest.approx(1.0)


# ─────────────────────────────── a folder of tables ──────────────────────────


def test_each_table_is_a_channel_named_after_its_stem(logger_dir):
    entries = _bare_channel_entries(logger_dir)
    assert entries == {
        name: {
            "kind": "raw",
            "loader": "csv",
            "directory": ".",
            "array_file": f"{name}.csv",
        }
        for name in ("1-baseline_accel", "1-baseline_orientations", "1-baseline_wrench")
    }


def test_init_declare_status_on_a_folder_of_tables(logger_dir, capsys):
    code, out = _cli(["init", str(logger_dir)], capsys)
    assert code in (0, None), out
    code, out = _cli(["check", str(logger_dir), "--json"], capsys)
    assert code == 1  # tables have no clock until it is declared

    _cli(["declare", str(logger_dir)], capsys)
    declaration = (logger_dir / "apairo.yaml").read_text()
    assert declaration.count("    key: {column: t, units: [us]}") == 3
    assert declaration.count('    directory: "."') == 3

    code, out = _cli(["check", str(logger_dir)], capsys)
    assert code == 0, out
    _, out = _cli(["status", str(logger_dir), "--json"], capsys)
    channels = json.loads(out)["channels"]
    assert channels["1-baseline_orientations"]["rate_hz"] == pytest.approx(100.0)
    assert channels["1-baseline_wrench"]["shape"] == [6]

    ds = RawDataset(logger_dir)
    view = ds.synchronize(reference="1-baseline_orientations", method="nearest")
    assert len(view) == 5
    assert view[0].data["1-baseline_wrench"].shape == (6,)


def test_a_lone_table_is_named_after_its_directory(tmp_path):
    d = tmp_path / "imu0"
    d.mkdir()
    _table(d / "data.csv", 5000, 4, 6)
    (d / "sensor.yaml").write_text("rate_hz: 200\n")
    assert _bare_channel_entries(d) == {
        "imu0": {
            "kind": "raw",
            "loader": "csv",
            "directory": ".",
            "array_file": "data.csv",
        }
    }


def test_frames_and_a_table_side_by_side(tmp_path):
    d = _bare_frames(tmp_path / "lidar")
    _table(d / "poses.csv", 100000, 5, 7)
    assert set(_bare_channel_entries(d)) == {"lidar", "poses"}


def test_only_numeric_text_files_are_tables(tmp_path):
    d = tmp_path / "logs"
    d.mkdir()
    (d / "groundtruth.txt").write_text("# timestamp tx ty\n1.0 2.0 3.0\n1.1 2.1 3.1\n")
    (d / "rgb.txt").write_text(
        "# color images\n1305031102.175304 rgb/1305031102.175304.png\n"
    )
    (d / "README.txt").write_text("Recorded on the rig, see notes.\n")
    (d / "index.csv").write_text(
        "#timestamp [ns],filename\n1403715273262142976,1403715273262142976.png\n"
    )
    assert set(_bare_channel_entries(d)) == {"logs"}  # groundtruth.txt alone
    assert _bare_channel_entries(d)["logs"]["array_file"] == "groundtruth.txt"


def test_an_index_csv_is_not_a_channel(tmp_path):
    """EuRoC's cam0/data.csv pairs a stamp with a filename: an index, not data."""
    cam = tmp_path / "cam0"
    (cam / "data").mkdir(parents=True)
    (cam / "data.csv").write_text(
        "#timestamp [ns],filename\n1403715273262142976,1403715273262142976.png\n"
    )
    assert _detect_loader(cam) is None


def test_init_merge_picks_up_a_new_table(logger_dir, capsys):
    _cli(["init", str(logger_dir)], capsys)
    _table(logger_dir / "2-vibrations_accel.csv", 3937, 6, 3)
    code, out = _cli(["init", str(logger_dir)], capsys)  # merge is the default
    assert code in (0, None), out
    assert "2-vibrations_accel" in read_config(logger_dir)["channels"]


# ───────────────────── a directory with a sub-directory is unchanged ──────────


def test_a_sequence_keeps_its_loose_tables_out(tmp_path):
    """Channels of a sequence are its sub-directories: a loose numeric table
    beside them (a KITTI-style poses.txt) is not auto-registered."""
    seq = tmp_path / "seq"
    _bare_frames(seq / "lidar")
    (seq / "poses.txt").write_text("0 0 0 1\n0 0 0 2\n")
    assert _bare_channel_entries(seq) == {}
    assert RawDataset(seq).keys == ["lidar"]


def test_a_root_is_still_a_root(tmp_path):
    root = tmp_path / "root"
    _bare_frames(root / "seq_a" / "lidar")
    _bare_frames(root / "seq_b" / "lidar")
    ds = RawDataset(root)
    assert ds._is_root
    assert ds.sequence_ids == ["seq_a", "seq_b"]


def test_declare_leaves_the_clock_column_as_homework_when_unsure(tmp_path, capsys):
    d = tmp_path / "log"
    d.mkdir()
    (d / "a.csv").write_text("x,y\n1,2\n3,4\n")  # no clock-like column
    (d / "b.csv").write_text("time,v\n0.0,1\n0.1,2\n0.2,3\n")  # relative seconds
    (d / "c.csv").write_text("t,v\n5,1\n3,2\n")  # a clock that goes backwards
    write_config(d, {"version": 1, "channels": {}})
    _, out = _cli(["declare", str(d), "-o", "-"], capsys)
    blocks = {b.split(":")[0]: b for b in re.split(r"\n  (?=\S)", out)[1:]}
    assert "# key: {column: <index or name>" in blocks["a"]
    assert (
        "    key: {column: time}   # clock column 'time', seconds assumed"
        in blocks["b"]
    )
    assert "# key: {column: <index or name>" in blocks["c"]
