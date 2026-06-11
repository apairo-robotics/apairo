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

from apairo import Rellis3DDataset, TartanKittiDataset

ASSETS = Path(__file__).parent.parent / "assets"


@pytest.fixture
def rellis_root(tmp_path):
    """Copy of the mini Rellis tree -- datasets write .apairo at first load."""
    dst = tmp_path / "mini_rellis"
    shutil.copytree(ASSETS / "mini_rellis", dst)
    return dst


@pytest.fixture
def tartan_seq(tmp_path):
    """Copy of the mini TartanDrive sequence."""
    dst = tmp_path / "figure_8"
    shutil.copytree(ASSETS / "mini_tartan" / "figure_8", dst)
    return dst


# ------------------------------------------------------------------ Rellis (sync)


def test_rellis_load(rellis_root):
    ds = Rellis3DDataset(rellis_root, keys=["lidar", "labels"])
    assert ds.is_synchronous
    assert len(ds) == 10  # 2 sequences x 5 frames

    sample = ds[0]
    assert sample.timestamp is None
    assert sample.data["lidar"].shape == (1024, 4)
    assert sample.data["lidar"].dtype == np.float32
    assert sample.data["labels"].shape == (1024,)
    assert sample.data["labels"].dtype == np.int64  # torch_dtype from profile


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
    view = (
        ds.transform("lidar", lambda pts: pts[pts[:, 0] > 0])
        .filter("labels", lambda lbl: len(np.unique(lbl)) > 3)
    )
    assert 0 < len(view) <= 10
    sample = view[0]
    assert (sample.data["lidar"][:, 0] > 0).all()


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
    sync = ds.synchronize(reference="velodyne_0", method="latest")
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


def test_tartan_synchronize_chain_shuffled_access(tartan_seq):
    ds = TartanKittiDataset(tartan_seq, keys=TARTAN_KEYS)
    view = (
        ds.synchronize(reference="velodyne_0", method="nearest", tolerance=0.15)
        .transform("velodyne_0", lambda pts: pts[pts[:, 2] > -1.0])
    )
    assert len(view) == 8
    for i in np.random.permutation(len(view)):
        sample = view[int(i)]
        assert set(sample.data) == set(TARTAN_KEYS)
        assert (sample.data["velodyne_0"][:, 2] > -1.0).all()
