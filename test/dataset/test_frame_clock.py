"""The synchronous frame clock.

A co-captured (synchronous) dataset carries a per-frame timestamp when any loaded
channel declares a ``key`` -- and because the frame is co-captured, that clock is
shared by every channel. ``is_synchronous`` stays True (structural), decoupled
from whether a clock is present. Multi-sequence: the flat clock resets per
sequence (each recording has its own timeline), so it is validated per sequence,
never globally.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from apairo.core.profiled_dataset import ProfiledDataset


class _KittiDS(ProfiledDataset):
    _profile = "semantic_kitti.yaml"


def _bin(path, n=8):
    np.random.rand(n, 4).astype(np.float32).tofile(path)


def _label(path, n=8):
    np.random.randint(0, 64, n, dtype=np.int32).tofile(path)


def _make_clocked_kitti(root, stamps, *, key_on="lidar"):
    """A kitti-layout dataset whose frames are named ``frame<i>-<sec>_<ms>`` so a
    timestamp lives in the filename; only *key_on* declares the clock key."""
    for seq, sts in stamps.items():
        (root / "sequences" / seq / "velodyne").mkdir(parents=True)
        (root / "sequences" / seq / "labels").mkdir(parents=True)
        for i, (s, ms) in enumerate(sts):
            stem = f"frame{i:06d}-{s}_{ms:03d}"
            _bin(root / "sequences" / seq / "velodyne" / f"{stem}.bin")
            _label(root / "sequences" / seq / "labels" / f"{stem}.label")
    channels: dict = {"lidar": {"loader": "bin"}, "labels": {"loader": "bin"}}
    if key_on:
        channels[key_on]["key"] = {
            "name": r"frame\d+-(\d+)_(\d+)",
            "units": ["s", "ms"],
        }
    (root / ".apairo").mkdir(exist_ok=True)
    (root / ".apairo" / "channels.yaml").write_text(
        yaml.safe_dump({"version": 1, "channels": channels})
    )


def test_synchronous_dataset_carries_a_shared_frame_clock(tmp_path):
    # seq 01's epoch is EARLIER than seq 00's -> the flat clock is NOT globally
    # monotonic; it resets per sequence. Each sequence is internally ascending.
    _make_clocked_kitti(
        tmp_path,
        {
            "00": [(100, 0), (100, 500), (101, 0)],
            "01": [(50, 0), (50, 500), (51, 0)],
        },
    )
    ds = _KittiDS(tmp_path, keys=["lidar", "labels"])

    assert ds.is_synchronous is True  # structural: co-captured frames
    assert ds.timestamps is not None  # ...yet clocked
    np.testing.assert_allclose(ds.timestamps, [100.0, 100.5, 101.0, 50.0, 50.5, 51.0])

    frame = ds[1]
    assert set(frame.data) == {"lidar", "labels"}  # a full synchronous frame
    assert frame.timestamp == pytest.approx(100.5)  # shared clock, from the lidar key
    assert ds[4].timestamp == pytest.approx(50.5)  # per-sequence, reset epoch


def test_clockless_when_no_channel_declares_a_key(tmp_path):
    _make_clocked_kitti(tmp_path, {"00": [(100, 0), (100, 500)]}, key_on=None)
    ds = _KittiDS(tmp_path, keys=["lidar", "labels"])
    assert ds.is_synchronous is True
    assert ds.timestamps is None
    assert ds[0].timestamp is None


def test_clock_comes_from_a_loaded_key_declaring_channel(tmp_path):
    # v1: the clock is derived from a LOADED channel that declares the key.
    # Loading only the un-keyed channel yields no clock (documented).
    _make_clocked_kitti(tmp_path, {"00": [(100, 0), (100, 500)]})
    ds = _KittiDS(tmp_path, keys=["labels"])
    assert ds.timestamps is None
    assert ds[0].timestamp is None


def test_frame_clock_rejects_a_non_monotonic_sequence(tmp_path):
    _make_clocked_kitti(tmp_path, {"00": [(100, 500), (100, 0)]})  # decreasing
    with pytest.raises(ValueError, match="non-decreasing within a sequence"):
        _KittiDS(tmp_path, keys=["lidar", "labels"])
