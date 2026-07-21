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


# -- the clock's origin is a concrete-dataset concern: a subclass provider ------


def test_clock_provider_escape_hatch(tmp_path):
    # No channel declares a key; the concrete dataset computes its own clock.
    _make_clocked_kitti(tmp_path, {"00": [(100, 0), (100, 500)]}, key_on=None)

    class _Provided(_KittiDS):
        def __init__(self, root, **kw):
            self._clock_provider = lambda ds: np.arange(len(ds), dtype=float) * 10.0
            super().__init__(root, **kw)

    ds = _Provided(tmp_path, keys=["lidar", "labels"])
    np.testing.assert_allclose(ds.timestamps, [0.0, 10.0])
    assert ds[1].timestamp == pytest.approx(10.0)


def test_clock_provider_takes_precedence_over_a_channel_key(tmp_path):
    _make_clocked_kitti(tmp_path, {"00": [(100, 0), (100, 500)]})  # lidar HAS a key

    class _Provided(_KittiDS):
        def __init__(self, root, **kw):
            self._clock_provider = lambda ds: np.array([7.0, 8.0])
            super().__init__(root, **kw)

    ds = _Provided(tmp_path, keys=["lidar", "labels"])
    np.testing.assert_allclose(ds.timestamps, [7.0, 8.0])  # provider wins


def test_clock_provider_wrong_length_raises(tmp_path):
    _make_clocked_kitti(tmp_path, {"00": [(100, 0), (100, 500)]}, key_on=None)

    class _Bad(_KittiDS):
        def __init__(self, root, **kw):
            self._clock_provider = lambda ds: np.array([1.0])  # 1 value for 2 frames
            super().__init__(root, **kw)

    with pytest.raises(ValueError, match="clock has 1 value"):
        _Bad(tmp_path, keys=["lidar", "labels"])


# -- profile clock: {file} -- a per-sequence sidecar (SemanticKITTI times.txt) --


def _make_kitti_with_times(root, stamps):
    for seq, ts in stamps.items():
        vel = root / "sequences" / seq / "velodyne"
        lab = root / "sequences" / seq / "labels"
        vel.mkdir(parents=True)
        lab.mkdir(parents=True)
        for i in range(len(ts)):
            _bin(vel / f"{i:06d}.bin")
            _label(lab / f"{i:06d}.label")
        (root / "sequences" / seq / "times.txt").write_text(
            "\n".join(str(t) for t in ts)
        )


def test_profile_clock_from_per_sequence_sidecar(tmp_path):
    # semantic_kitti.yaml declares `clock: {file: times.txt}`. Positional frames,
    # each sequence's times.txt is the shared clock (distinct epochs per sequence).
    _make_kitti_with_times(
        tmp_path, {"00": [0.0, 0.1, 0.2, 0.3], "01": [10.0, 10.1, 10.2, 10.3]}
    )
    ds = _KittiDS(tmp_path, keys=["lidar", "labels"])
    assert ds.is_synchronous
    np.testing.assert_allclose(
        ds.timestamps, [0.0, 0.1, 0.2, 0.3, 10.0, 10.1, 10.2, 10.3]
    )
    assert ds[5].timestamp == pytest.approx(10.1)


def test_profile_clock_sidecar_absent_is_clockless(tmp_path):
    # A declared clock: whose sidecar is absent stays clockless -- not an error.
    vel = tmp_path / "sequences" / "00" / "velodyne"
    lab = tmp_path / "sequences" / "00" / "labels"
    vel.mkdir(parents=True)
    lab.mkdir(parents=True)
    for i in range(2):
        _bin(vel / f"{i:06d}.bin")
        _label(lab / f"{i:06d}.label")
    ds = _KittiDS(tmp_path, keys=["lidar", "labels"])
    assert ds.timestamps is None
    assert ds[0].timestamp is None
