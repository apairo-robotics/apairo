"""Smoke tests on real extracted data (test/assets/).

These fixtures are genuine excerpts from Rellis-3D (synchronous) and
TartanDrive v2 (asynchronous), subsampled by ``test/assets/extract_mini_datasets.py``.
They exercise the full path -- file discovery, profiles, loaders, .apairo
bootstrap, splits, timeline, synchronize -- on data the synthetic tests
cannot fake.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from apairo import (
    FramePreprocessor,
    Rellis3DDataset,
    SequencePreprocessor,
    TartanKittiDataset,
)
from apairo.core.config import register_static_transform

ASSETS = Path(__file__).parent.parent / "assets"


@pytest.fixture
def rellis_root(tmp_path):
    """Copy of the mini Rellis tree -- datasets write .apairo at first load."""
    dst = tmp_path / "mini_rellis"
    shutil.copytree(ASSETS / "mini_rellis", dst)
    return dst


@pytest.fixture
def tartan_seq(tmp_path):
    """Copy of the mini TartanDrive sequence -- always bare, so the auto
    bootstrap runs even when a stray local run left a sidecar in the assets."""
    dst = tmp_path / "figure_8"
    shutil.copytree(
        ASSETS / "mini_tartan" / "figure_8",
        dst,
        ignore=shutil.ignore_patterns(".apairo"),
    )
    return dst


# ------------------------------------------------------------------ Rellis (sync)


def test_rellis_load(rellis_root):
    ds = Rellis3DDataset(rellis_root, keys=["lidar", "labels"])
    assert ds.is_synchronous
    assert len(ds) == 10  # 2 sequences x 5 frames

    sample = ds[0]
    # Rellis is synchronous AND clocked: lidar/labels are positional, but each is
    # co-captured with a timestamped camera frame, so the camera IS the shared
    # frame clock (profile `clock:`) -- present even though the camera is not
    # among the loaded keys.
    assert sample.timestamp is not None
    assert "pylon_camera_node" not in sample.data  # clock without loading the source
    assert ds.timestamps is not None and len(ds.timestamps) == 10
    assert sample.data["lidar"].shape == (1024, 4)
    assert sample.data["lidar"].dtype == np.float32
    assert sample.data["labels"].shape == (1024,)
    assert sample.data["labels"].dtype == np.int64  # cast_dtype from profile


def test_rellis_frame_clock(rellis_root):
    # The camera (profile `clock:`) provides the shared frame clock without being
    # loaded as a channel. Each sequence carries its own epoch, and the clock
    # realigns to the SELECTED frames under a split.
    ds = Rellis3DDataset(rellis_root, keys=["lidar", "labels"])
    assert ds.timestamps is not None and len(ds.timestamps) == 10
    assert list(ds.timestamps[:5]) == sorted(ds.timestamps[:5])  # seq 00000 ascends
    assert list(ds.timestamps[5:]) == sorted(ds.timestamps[5:])  # seq 00001 ascends
    assert ds[0].timestamp != ds[5].timestamp  # distinct per-sequence epochs

    train = Rellis3DDataset(rellis_root, keys=["lidar", "labels"], split="train")
    assert len(train.timestamps) == len(train)
    full = set(np.round(ds.timestamps, 3))
    assert set(np.round(train.timestamps, 3)).issubset(full)  # realigned subset


def test_rellis_poses(rellis_root):
    ds = Rellis3DDataset(rellis_root, keys=["lidar", "labels", "poses"])
    assert ds[0].data["poses"].shape == (3, 4)


def test_rellis_splits(rellis_root):
    keys = ["lidar", "labels"]
    assert len(Rellis3DDataset(rellis_root, keys=keys, split="train")) == 5
    assert len(Rellis3DDataset(rellis_root, keys=keys, split="val")) == 3
    assert len(Rellis3DDataset(rellis_root, keys=keys, split="test")) == 2


def test_rellis_sequences(rellis_root):
    ds = Rellis3DDataset(rellis_root, keys=["lidar", "labels"])
    assert ds.sequence_ids == ["00000", "00001"]
    assert len(ds.filter_sequences(["00001"])) == 5
    assert len(ds.sequence("00000")) == 5


def test_rellis_chaining(rellis_root):
    ds = Rellis3DDataset(rellis_root, keys=["lidar", "labels"])
    view = ds.transform("lidar", lambda pts: pts[pts[:, 0] > 0]).filter(
        "labels", lambda lbl: len(np.unique(lbl)) > 3
    )
    assert 0 < len(view) <= 10
    sample = view[0]
    assert (sample.data["lidar"][:, 0] > 0).all()


def test_rellis_derived_channel_split(rellis_root):
    """A preprocessed (derived) channel loads correctly under an lst split.

    Regression: _discover_derived used to apply the path-based split filter
    unconditionally, so a derived channel (no split dir in its path) came back
    empty under split="train" on lst-split datasets like Rellis.
    """

    class TravLabel(FramePreprocessor):
        output_key = "trav_gt"
        output_loader = "npys"
        input_keys = ["labels"]
        timestamps_from = "lidar"
        sources = ["labels"]

        def __call__(self, sample):
            return (sample.data["labels"] > 0).astype(np.uint8)

    Rellis3DDataset(rellis_root, keys=["lidar", "labels"]).run_preprocess(TravLabel())

    ds = Rellis3DDataset(rellis_root, keys=["lidar", "trav_gt"], split="train")
    assert len(ds) == 5  # train split, not the full 10 frames
    sample = ds[0]
    assert sample.data["trav_gt"].shape == sample.data["lidar"].shape[:1]

    # filter_split() reaches the same frames from the unsplit dataset.
    full = Rellis3DDataset(rellis_root, keys=["lidar", "trav_gt"])
    assert len(full.filter_split("train")) == 5


def test_rellis_poses_under_split(rellis_root):
    """A stacked sequence-file channel (poses, txt_rows) loads frame-aligned under
    a split. Regression: it kept every row regardless of the split and raised
    'Mismatched frame counts per key'.
    """
    full = Rellis3DDataset(rellis_root, keys=["lidar", "poses"])
    gt = {
        (full.frame_sequence_ids[i], full.frame_stems[i]): full[i].data["poses"]
        for i in range(len(full))
    }
    for split, n in [("train", 5), ("val", 3), ("test", 2)]:
        ds = Rellis3DDataset(rellis_root, keys=["lidar", "poses"], split=split)
        assert len(ds) == n
        for i in range(len(ds)):
            key = (ds.frame_sequence_ids[i], ds.frame_stems[i])
            assert ds[i].data["poses"].shape == (3, 4)
            np.testing.assert_allclose(ds[i].data["poses"], gt[key])


def test_rellis_calibration_resolves(rellis_root):
    """A synchronous ProfiledDataset exposes calibration from its root .apairo --
    not just RawDataset. Regression: ds.calibration used to always be empty unless
    the dataset was a RawDataset."""
    T = np.eye(4)
    T[:3, 3] = [0.0, 0.0, 1.0]  # os1_lidar mounted 1 m above base_link
    register_static_transform(rellis_root, "base_link", "os1_lidar", T)

    ds = Rellis3DDataset(rellis_root, keys=["lidar", "labels"])
    assert "base_link_to_os1_lidar" in ds.calibration
    np.testing.assert_allclose(
        ds.calibration.get_tf("os1_lidar", "base_link")[:3, 3], [0, 0, 1.0]
    )


def test_sequence_preprocessor_per_frame_multi_sequence(rellis_root):
    """A per-frame SequencePreprocessor (output_loader='npys') runs once per
    sequence and writes one file per frame, so a multi-sequence ProfiledDataset
    loads it back. Regression: it used to write a single root-level stacked file,
    invisible to per-sequence discovery (and crossing sequence boundaries).
    """

    class PositionInSequence(SequencePreprocessor):
        # Emit each frame's index within its own sequence -- a sequence-global
        # computation whose per-frame output resets to 0 at every boundary.
        output_key = "seq_pos"
        output_loader = "npys"
        input_keys = ["lidar"]
        timestamps_from = "lidar"
        sources = ["lidar"]

        def __call__(self, frames):
            return np.arange(len(list(frames)), dtype=np.int64)

    Rellis3DDataset(rellis_root, keys=["lidar"]).run_preprocess(PositionInSequence())

    ds = Rellis3DDataset(rellis_root, keys=["lidar", "seq_pos"])
    assert len(ds) == 10
    pos = np.array([int(ds[i].data["seq_pos"]) for i in range(len(ds))])
    # Per-sequence reset proves process() ran per sequence, not across the root.
    np.testing.assert_array_equal(pos, [0, 1, 2, 3, 4, 0, 1, 2, 3, 4])

    # And it loads under a split, frame-aligned with lidar.
    train = Rellis3DDataset(rellis_root, keys=["lidar", "seq_pos"], split="train")
    assert len(train) == 5


def test_sequence_preprocessor_stacked_multi_sequence(rellis_root):
    """A stacked SequencePreprocessor (output_loader='npy') writes one stacked
    file per sequence and loads back frame-aligned on a multi-sequence dataset,
    including under a split -- the NPYLoader-style stacked path on ProfiledDataset.
    """

    class StackedPose(SequencePreprocessor):
        output_key = "icp_pose"
        output_loader = "npy"  # one stacked <seq>/icp_pose/icp_pose.npy per sequence
        input_keys = ["lidar"]
        timestamps_from = "lidar"
        sources = ["lidar"]

        def __call__(self, frames):
            n = len(list(frames))
            return np.stack([np.eye(4) * i for i in range(n)])

    Rellis3DDataset(rellis_root, keys=["lidar"]).run_preprocess(StackedPose())

    ds = Rellis3DDataset(rellis_root, keys=["lidar", "icp_pose"])
    assert len(ds) == 10
    assert ds[0].data["icp_pose"].shape == (4, 4)
    diag = np.array([ds[i].data["icp_pose"][0, 0] for i in range(len(ds))])
    np.testing.assert_array_equal(diag, [0, 1, 2, 3, 4, 0, 1, 2, 3, 4])

    train = Rellis3DDataset(rellis_root, keys=["lidar", "icp_pose"], split="train")
    assert len(train) == 5
    assert train[0].data["icp_pose"].shape == (4, 4)


# ------------------------------------------------------------------ Tartan (async)

TARTAN_KEYS = ["velodyne_0", "cmd", "multisense_imu"]


def test_tartan_auto_bootstrap(tartan_seq):
    ds = TartanKittiDataset(tartan_seq, keys=TARTAN_KEYS)
    assert (tartan_seq / ".apairo").exists()
    assert not ds.is_synchronous
    # 8 velodyne + 9 cmd + 365 imu events
    assert len(ds) == 8 + 9 + 365


def test_tartan_timeline_order_and_events(tartan_seq):
    ds = TartanKittiDataset(tartan_seq, keys=TARTAN_KEYS)
    last = -np.inf
    seen = set()
    for sample in ds:
        assert len(sample.data) == 1  # one event, one channel
        assert sample.timestamp >= last
        last = sample.timestamp
        seen.update(sample.data)
    assert seen == set(TARTAN_KEYS)


def test_tartan_synchronize(tartan_seq):
    ds = TartanKittiDataset(tartan_seq, keys=TARTAN_KEYS)
    sync = ds.synchronize(reference="velodyne_0", method="previous")
    assert sync.is_synchronous
    assert len(sync) == 8  # every channel has an event before frame 0

    sample = sync[0]
    assert set(sample.data) == set(TARTAN_KEYS)
    assert sample.data["velodyne_0"].shape == (512, 3)
    assert sample.data["cmd"].shape == (2,)
    assert sample.data["multisense_imu"].shape == (6,)

    # latest: matched events are never in the future of the reference clock
    assert (sync.time_offsets("multisense_imu") <= 0).all()
    # imu fires at ~400 Hz: the latest event is always fresh
    assert abs(sync.time_offsets("multisense_imu")).max() < 0.01


def test_tartan_calibration_resolves(tartan_seq):
    """An asynchronous profiled dataset (TartanKittiDataset) reads calibration from
    its sequence .apairo, same as RawDataset -- the static tree is resolvable
    without dropping to the generic loader."""
    T = np.eye(4)
    T[:3, 3] = [0.0, 0.0, 1.2]  # velodyne 1.2 m above base_link
    register_static_transform(tartan_seq, "base_link", "velodyne", T)

    ds = TartanKittiDataset(tartan_seq, keys=["velodyne_0"])
    np.testing.assert_allclose(
        ds.calibration.get_tf("velodyne", "base_link")[:3, 3], [0, 0, 1.2]
    )


def test_tartan_velodyne_intensity_auto_discovered(tartan_seq):
    """velodyne_0/000000_intensity.npy is a suffixed sub-channel, auto-discovered
    as velodyne_0_intensity -- sharing velodyne_0's directory and clock."""
    ds = TartanKittiDataset(tartan_seq, keys=["velodyne_0", "velodyne_0_intensity"])
    np.testing.assert_array_equal(
        ds.timestamps["velodyne_0"], ds.timestamps["velodyne_0_intensity"]
    )

    sync = ds.synchronize(reference="velodyne_0", method="previous")
    sample = sync[0]
    expected = np.load(tartan_seq / "velodyne_0" / "000000_intensity.npy")
    np.testing.assert_allclose(sample.data["velodyne_0_intensity"], expected)
    assert sample.data["velodyne_0_intensity"].shape == (512,)
    assert sample.data["velodyne_0"].shape == (512, 3)


def test_tartan_synchronize_chain_shuffled_access(tartan_seq):
    ds = TartanKittiDataset(tartan_seq, keys=TARTAN_KEYS)
    view = ds.synchronize(
        reference="velodyne_0", method="nearest", tolerance=0.15
    ).transform("velodyne_0", lambda pts: pts[pts[:, 2] > -1.0])
    assert len(view) == 8
    for i in np.random.permutation(len(view)):
        sample = view[int(i)]
        assert set(sample.data) == set(TARTAN_KEYS)
        assert (sample.data["velodyne_0"][:, 2] > -1.0).all()
