"""A sensor's transport delay, declared once: ``latency``.

A frame stamped when it reached the computer was captured earlier -- a sensor
read over a bus and forwarded by a microcontroller arrives late, by a fixed
delay its authors may document (8416 us for the IMU of the KUKA F/T logs). Its
timestamps alone cannot reveal it: only the data can, and only a declaration
can correct it. ``latency: <seconds>`` moves the channel's clock back by that
much, whatever the clock's source, before anything is aligned to it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from apairo.cli import main
from apairo.core.config import verify_config, write_config
from apairo.dataset.raw import RawDataset

T0 = 100.0


def _frames(seq, name, stamps, latency=None, **extra):
    d = seq / name
    d.mkdir(parents=True)
    for i in range(len(stamps)):
        np.save(d / f"{i:06d}.npy", np.full(2, i, np.float32))
    np.savetxt(d / "timestamps.txt", stamps, fmt="%.6f")
    entry = {"kind": "raw", "loader": "npys", **extra}
    if latency is not None:
        entry["latency"] = latency
    return entry


def _seq(tmp_path, **channels):
    seq = tmp_path / "seq"
    entries = {name: _frames(seq, name, *spec) for name, spec in channels.items()}
    write_config(seq, {"version": 1, "channels": entries})
    return seq


def test_a_declared_latency_moves_the_clock_back(tmp_path):
    seq = _seq(tmp_path, imu=(T0 + np.arange(4) * 0.01, 0.0084))
    np.testing.assert_allclose(
        RawDataset(seq).timestamps["imu"], T0 - 0.0084 + np.arange(4) * 0.01
    )


def test_a_table_clock_is_corrected_too(tmp_path):
    seq = tmp_path / "seq"
    seq.mkdir()
    (seq / "imu.csv").write_text("t,a\n1000000,1\n1010000,2\n1020000,3\n")
    write_config(
        seq,
        {
            "version": 1,
            "channels": {
                "imu": {
                    "kind": "raw",
                    "loader": "csv",
                    "directory": ".",
                    "array_file": "imu.csv",
                    "key": {"column": "t", "units": ["us"]},
                    "latency": 0.008416,
                }
            },
        },
    )
    np.testing.assert_allclose(
        RawDataset(seq).timestamps["imu"], [0.991584, 1.001584, 1.011584]
    )


def test_a_borrowed_clock_carries_the_source_latency_and_its_own(tmp_path):
    seq = tmp_path / "seq"
    lidar = _frames(seq, "lidar", T0 + np.arange(3) * 0.1, 0.02)
    labels = _frames(seq, "labels", T0 + np.arange(3) * 0.1, 0.005)
    (seq / "labels" / "timestamps.txt").unlink()
    labels["timestamps_from"] = "lidar"
    write_config(seq, {"version": 1, "channels": {"lidar": lidar, "labels": labels}})
    ds = RawDataset(seq)
    np.testing.assert_allclose(ds.timestamps["lidar"], T0 - 0.02 + np.arange(3) * 0.1)
    np.testing.assert_allclose(ds.timestamps["labels"], T0 - 0.025 + np.arange(3) * 0.1)


def test_synchronize_pairs_on_the_corrected_clock(tmp_path):
    """IMU frame 10 is captured with camera frame 1 but stamped 8 ms after it,
    so nearest matching picks frame 9 -- stamped 2 ms before the camera,
    captured 10 ms before it -- until the latency is declared."""
    cam = T0 + np.arange(3) * 0.1
    imu = T0 + np.arange(31) * 0.01 + 0.008
    for latency, expected in ((None, 9), (0.008, 10)):
        seq = _seq(tmp_path / str(latency), cam=(cam,), imu=(imu, latency))
        view = RawDataset(seq).synchronize(reference="cam", method="nearest")
        assert view.frame_indices["imu"][1] == expected


@pytest.mark.parametrize("bad", ["8 ms", True, float("nan")])
def test_check_wants_a_number_of_seconds(tmp_path, bad):
    seq = _seq(tmp_path, imu=(T0 + np.arange(2) * 0.01, bad))
    assert any("'latency' is a number of seconds" in i for i in verify_config(seq))


def test_status_shows_the_corrected_span(tmp_path, capsys):
    seq = _seq(tmp_path, imu=(T0 + np.arange(5) * 0.01, 0.5))
    try:
        main(["status", str(seq), "--json"])
    except SystemExit:
        pass
    span = json.loads(capsys.readouterr().out)["channels"]["imu"]["span"]
    assert span == pytest.approx([T0 - 0.5, T0 - 0.5 + 0.04])


@pytest.mark.parametrize("command", ["status", "check"])
def test_a_latency_that_is_not_a_number_is_reported_not_a_crash(
    tmp_path, capsys, command
):
    seq = _seq(tmp_path, imu=(T0 + np.arange(3) * 0.01, "8 ms"))
    try:
        code = main([command, str(seq)])
    except SystemExit as exc:
        code = exc.code
    out = capsys.readouterr().out
    assert code == 1 if command == "check" else code in (0, None)
    assert "'latency' is a number of seconds" in out
    with pytest.raises(
        ValueError, match="Channel 'imu': 'latency' is a number of seconds"
    ):
        RawDataset(seq)


def test_yaml_s_8e_3_is_a_number(tmp_path):
    """YAML 1.1 reads `8e-3` as a string; it is the number it looks like."""
    seq = _seq(tmp_path, imu=(T0 + np.arange(3) * 0.01,))
    (seq / "apairo.yaml").write_text("version: 1\nchannels:\n  imu: {latency: 8e-3}\n")
    assert verify_config(seq) == []
    np.testing.assert_allclose(RawDataset(seq).timestamps["imu"][0], T0 - 0.008)


def test_a_suffixed_sub_channel_shares_its_directory_s_latency(tmp_path):
    seq = tmp_path / "seq"
    d = seq / "velodyne_0"
    d.mkdir(parents=True)
    for i in range(4):
        np.save(d / f"{i:06d}.npy", np.zeros((5, 3), np.float32))
        np.save(d / f"{i:06d}_intensity.npy", np.zeros(5, np.uint8))
    np.savetxt(d / "timestamps.txt", T0 + np.arange(4) * 0.1)
    write_config(
        seq,
        {
            "version": 1,
            "channels": {
                "velodyne_0": {"kind": "raw", "loader": "npys", "latency": 0.05},
                "velodyne_0_intensity": {
                    "kind": "raw",
                    "loader": "npys",
                    "directory": "velodyne_0",
                    "suffix": "intensity",
                },
            },
        },
    )
    ds = RawDataset(seq)
    np.testing.assert_allclose(
        ds.timestamps["velodyne_0_intensity"], ds.timestamps["velodyne_0"]
    )
    view = ds.synchronize(reference="velodyne_0", method="nearest")
    np.testing.assert_array_equal(
        view.frame_indices["velodyne_0_intensity"], view.frame_indices["velodyne_0"]
    )


def test_a_timestamp_that_is_not_a_number_is_refused(tmp_path, capsys):
    seq = _seq(tmp_path, imu=(np.array([T0, np.nan, T0 + 0.02]),))
    with pytest.raises(
        ValueError, match="Channel 'imu': timestamp 1 is not a finite number"
    ):
        RawDataset(seq)
    try:
        code = main(["check", str(seq)])
    except SystemExit as exc:
        code = exc.code
    assert code == 1 and "not a finite number" in capsys.readouterr().out


def test_a_table_clock_that_is_not_a_number_is_refused(tmp_path):
    seq = tmp_path / "seq"
    seq.mkdir()
    (seq / "imu.csv").write_text("t,a\n0.00,1\nnan,2\n0.20,3\n")
    write_config(
        seq,
        {
            "version": 1,
            "channels": {
                "imu": {
                    "kind": "raw",
                    "loader": "csv",
                    "directory": ".",
                    "array_file": "imu.csv",
                    "key": {"column": "t"},
                }
            },
        },
    )
    with pytest.raises(
        ValueError, match="Channel .imu.: timestamp 1 is not a finite number"
    ):
        RawDataset(seq)
