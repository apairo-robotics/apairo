"""A channel's clock must be ascending whatever its origin: a timestamps.txt
is checked like a declared key."""

from __future__ import annotations

import numpy as np
import pytest

from apairo import RawDataset


def _channel(root, name, stamps):
    d = root / name
    d.mkdir(parents=True)
    for i in range(len(stamps)):
        np.save(d / f"{i:06d}.npy", np.full(2, float(i)))
    np.savetxt(d / "timestamps.txt", stamps, fmt="%.6f")


def test_an_unsorted_timestamps_file_is_refused(tmp_path):
    _channel(tmp_path, "lidar", [0.0, 0.1, 0.2])
    _channel(tmp_path, "pose", [0.0, 0.9, 0.1])  # frame 2 goes back in time
    with pytest.raises(
        ValueError, match=r"'pose'.*not non-decreasing.*frame 2 is at 0\.1"
    ):
        RawDataset(tmp_path, keys=["lidar", "pose"])


def test_equal_consecutive_times_are_accepted(tmp_path):
    _channel(tmp_path, "lidar", [0.0, 0.1, 0.1, 0.2])
    assert len(RawDataset(tmp_path, keys=["lidar"])) == 4
