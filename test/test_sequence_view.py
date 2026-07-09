from pathlib import Path

import numpy as np
import pytest

from apairo.dataset.rellis import Rellis3DDataset

N_POINTS = 60


def _make_bin(path: Path):
    np.random.rand(N_POINTS, 4).astype(np.float32).tofile(path)


def _make_label(path: Path):
    np.random.randint(0, 20, N_POINTS, dtype=np.int32).tofile(path)


@pytest.fixture
def rellis_root(tmp_path):
    n_frames = 4
    for seq in ["00000", "00001", "00002"]:
        bin_dir = tmp_path / "Rellis-3D" / seq / "os1_cloud_node_kitti_bin"
        lbl_dir = tmp_path / "Rellis-3D" / seq / "os1_cloud_node_semantickitti_label_id"
        bin_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        for i in range(n_frames):
            _make_bin(bin_dir / f"{i:06d}.bin")
            _make_label(lbl_dir / f"{i:06d}.label")
    return tmp_path, n_frames


def test_sequence_ids(rellis_root):
    root, _ = rellis_root
    ds = Rellis3DDataset(root, keys=["lidar"])
    assert ds.sequence_ids == ["00000", "00001", "00002"]


def test_sequence_view_len(rellis_root):
    root, n_frames = rellis_root
    ds = Rellis3DDataset(root, keys=["lidar"])
    seq = ds.sequence("00001")
    assert len(seq) == n_frames


def test_sequence_view_getitem(rellis_root):
    root, n_frames = rellis_root
    ds = Rellis3DDataset(root, keys=["lidar"])
    seq = ds.sequence("00000")
    sample = seq[0]
    assert sample.data["lidar"].shape == (N_POINTS, 4)


def test_sequence_view_iter(rellis_root):
    root, n_frames = rellis_root
    ds = Rellis3DDataset(root, keys=["lidar"])
    seq = ds.sequence("00002")
    assert len(list(seq)) == n_frames


def test_sequence_view_sequence_id(rellis_root):
    root, _ = rellis_root
    ds = Rellis3DDataset(root, keys=["lidar"])
    seq = ds.sequence("00001")
    assert seq.sequence_id == "00001"


def test_sequence_view_repr(rellis_root):
    root, n_frames = rellis_root
    ds = Rellis3DDataset(root, keys=["lidar"])
    seq = ds.sequence("00000")
    assert "00000" in repr(seq)
    assert str(n_frames) in repr(seq)


def test_tuple_getitem(rellis_root):
    root, _ = rellis_root
    ds = Rellis3DDataset(root, keys=["lidar"])
    assert ds["00000", 0].data["lidar"].shape == (N_POINTS, 4)
    assert ds["00001", 2].data["lidar"].shape == (N_POINTS, 4)


def test_tuple_getitem_equals_sequence_view(rellis_root):
    root, n_frames = rellis_root
    ds = Rellis3DDataset(root, keys=["lidar"])
    for i in range(n_frames):
        direct = ds["00000", i].data["lidar"]
        via_view = ds.sequence("00000")[i].data["lidar"]
        np.testing.assert_array_equal(direct, via_view)


def test_sequence_view_out_of_range(rellis_root):
    root, n_frames = rellis_root
    ds = Rellis3DDataset(root, keys=["lidar"])
    seq = ds.sequence("00000")
    with pytest.raises(IndexError):
        seq[n_frames]


def test_unknown_sequence_raises(rellis_root):
    root, _ = rellis_root
    ds = Rellis3DDataset(root, keys=["lidar"])
    with pytest.raises(KeyError):
        ds.sequence("99999")


def test_sequences_cover_all_frames(rellis_root):
    root, n_frames = rellis_root
    ds = Rellis3DDataset(root, keys=["lidar"])
    total = sum(len(ds.sequence(sid)) for sid in ds.sequence_ids)
    assert total == len(ds)
