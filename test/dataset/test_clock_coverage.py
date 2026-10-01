"""A channel's clock must hold exactly one timestamp per frame.

The timeline gives a channel one slot per timestamp, so a mismatch used to be
wrong in both directions: a short ``timestamps.txt`` silently dropped the
trailing frames (a gap in the middle shifted every later frame onto the wrong
timestamp), and a long one -- or a ``timestamps_from`` borrowed from a channel
with more frames -- built fine and failed later with a bare ``IndexError``.
Loading now refuses at construction, naming both counts, and ``check`` reports
the same thing without loading anything.
"""

import json

import numpy as np
import pytest

from apairo.cli import main
from apairo.core.config import write_config
from apairo.dataset.raw import RawDataset


def _channel(root, name, n_frames, n_stamps=None):
    d = root / name
    d.mkdir(parents=True)
    for i in range(n_frames):
        np.save(d / f"{i:06d}.npy", np.full(3, i, np.float32))
    if n_stamps is not None:
        np.savetxt(d / "timestamps.txt", np.arange(n_stamps) * 0.1)


def _config(root, **channels):
    write_config(
        root,
        {
            "version": 1,
            "channels": {
                name: {"kind": "raw", "loader": "npys", **extra}
                for name, extra in channels.items()
            },
        },
    )


def test_matching_clock_loads(tmp_path):
    _channel(tmp_path, "lidar", 5, 5)
    _config(tmp_path, lidar={})
    assert len(RawDataset(tmp_path, keys=["lidar"])) == 5


@pytest.mark.parametrize("n_frames, n_stamps", [(6, 5), (5, 7)])
def test_own_timestamps_that_do_not_match_the_frames_are_refused(
    tmp_path, n_frames, n_stamps
):
    _channel(tmp_path, "lidar", n_frames, n_stamps)
    _config(tmp_path, lidar={})
    with pytest.raises(
        ValueError,
        match=rf"'lidar' has {n_frames} frame\(s\) but {n_stamps} timestamp\(s\) "
        r"in 'lidar/timestamps.txt'",
    ):
        RawDataset(tmp_path, keys=["lidar"])


def test_a_borrowed_clock_must_match_frame_for_frame(tmp_path):
    _channel(tmp_path, "lidar", 46, 46)
    _channel(tmp_path, "trav", 45)
    _config(tmp_path, lidar={}, trav={"timestamps_from": "lidar"})
    with pytest.raises(
        ValueError,
        match=r"'trav' has 45 frame\(s\) but borrows the clock of 'lidar' "
        r"\(timestamps_from\), which has 46",
    ):
        RawDataset(tmp_path, keys=["lidar", "trav"])


def test_a_borrowed_clock_that_matches_still_loads(tmp_path):
    _channel(tmp_path, "lidar", 4, 4)
    _channel(tmp_path, "trav", 4)
    _config(tmp_path, lidar={}, trav={"timestamps_from": "lidar"})
    ds = RawDataset(tmp_path, keys=["lidar", "trav"])
    assert len(ds) == 8
    np.testing.assert_array_equal(ds.timestamps["trav"], ds.timestamps["lidar"])


def _check(path, capsys):
    try:
        code = main(["check", str(path), "--json"])
    except SystemExit as exc:
        code = exc.code
    return code, json.loads(capsys.readouterr().out)


def test_check_reports_clock_mismatches_without_loading(tmp_path, capsys):
    _channel(tmp_path, "lidar", 46, 46)
    _channel(tmp_path, "trav", 45)
    _channel(tmp_path, "camera", 6, 5)
    _config(tmp_path, lidar={}, trav={"timestamps_from": "lidar"}, camera={})
    code, out = _check(tmp_path, capsys)
    assert code == 1
    issues = "\n".join(out["issues"])
    assert "channel 'camera': 6 frame(s) but 5 timestamp(s)" in issues
    assert "channel 'trav': 45 frame(s) but 46 timestamp(s)" in issues
    assert not any(i.startswith("channel 'lidar'") for i in out["issues"])


def test_check_reports_a_channel_with_no_clock(tmp_path, capsys):
    _channel(tmp_path, "scans", 3)
    _config(tmp_path, scans={})
    code, out = _check(tmp_path, capsys)
    assert code == 1
    assert any("channel 'scans': no clock" in i for i in out["issues"])


def test_check_leaves_a_declared_key_to_loading(tmp_path, capsys):
    d = tmp_path / "camera"
    d.mkdir()
    for t in (100, 200):
        np.save(d / f"{t}.npy", np.zeros(3, np.float32))
    _config(tmp_path, camera={"key": {"name": r"(\d+)", "units": ["ms"]}})
    code, out = _check(tmp_path, capsys)
    assert code == 0, out["issues"]


def test_check_reads_the_layout_through_an_external_declaration(tmp_path, capsys):
    """A key declared in an external file (``--declare``, ``declare=`` at load
    time) is a clock: ``check`` must not call the channel clockless, and
    ``status`` must show what loading with that declaration sees."""
    d = tmp_path / "seq" / "camera"
    d.mkdir(parents=True)
    for t in (100, 200, 300):
        np.save(d / f"{t}.npy", np.zeros(3, np.float32))
    _config(tmp_path / "seq", camera={})
    declaration = tmp_path / "outside.yaml"
    declaration.write_text(
        "version: 1\n"
        "channels:\n"
        "  camera: {loader: npys, key: {name: '(\\d+)', units: [ms]}}\n"
    )

    code, out = _check(tmp_path / "seq", capsys)
    assert code == 1
    assert any("channel 'camera': no clock" in i for i in out["issues"])

    try:
        code = main(
            ["check", str(tmp_path / "seq"), "--declare", str(declaration), "--json"]
        )
    except SystemExit as exc:
        code = exc.code
    assert code == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True, "issues": []}
    assert len(RawDataset(tmp_path / "seq", declare=declaration)) == 3
