"""A channel's clock must hold exactly one timestamp per frame.

The timeline gives a channel one slot per timestamp, so a mismatch used to be
wrong in both directions: a short ``timestamps.txt`` silently dropped the
trailing frames (a gap in the middle shifted every later frame onto the wrong
timestamp), and a long one -- or a ``timestamps_from`` borrowed from a channel
with more frames -- built fine and failed later with a bare ``IndexError``.
Loading now refuses at construction, naming both counts.
"""

import numpy as np
import pytest

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
