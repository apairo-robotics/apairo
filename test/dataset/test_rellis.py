import pytest
import numpy as np
from pathlib import Path

from apairo.dataset.rellis import Rellis3DDataset
from apairo.core.sample import Sample

N_POINTS = 60


def _make_bin(path: Path, n: int = N_POINTS):
    np.random.rand(n, 4).astype(np.float32).tofile(path)


def _make_label(path: Path, n: int = N_POINTS):
    np.random.randint(0, 20, n, dtype=np.int32).tofile(path)


def _make_poses(path: Path, n_frames: int = 3):
    np.savetxt(path, np.random.rand(n_frames, 12))


@pytest.fixture
def rellis_root(tmp_path):
    n_frames = 3
    for seq in ["00000", "00001"]:
        bin_dir = tmp_path / "Rellis-3D" / seq / "os1_cloud_node_kitti_bin"
        lbl_dir = tmp_path / "Rellis-3D" / seq / "os1_cloud_node_semantickitti_label_id"
        bin_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        for i in range(n_frames):
            _make_bin(bin_dir / f"{i:06d}.bin")
            _make_label(lbl_dir / f"{i:06d}.label")
    return tmp_path, n_frames * 2  # 2 sequences x n_frames


def test_len(rellis_root):
    root, total = rellis_root
    ds = Rellis3DDataset(root, keys=["lidar", "labels"])
    assert len(ds) == total


def test_getitem_returns_sample(rellis_root):
    root, _ = rellis_root
    ds = Rellis3DDataset(root, keys=["lidar", "labels"])
    s = ds[0]
    assert isinstance(s, Sample)
    assert s.timestamp is None
    assert s.data["lidar"].shape == (N_POINTS, 4)
    assert s.data["labels"].shape == (N_POINTS,)
    assert s.data["lidar"].dtype == np.float32
    assert s.data["labels"].dtype == np.int64


def test_iter(rellis_root):
    root, total = rellis_root
    ds = Rellis3DDataset(root, keys=["lidar", "labels"])
    assert len(list(ds)) == total


def test_next(rellis_root):
    root, _ = rellis_root
    ds = Rellis3DDataset(root, keys=["lidar"])
    it = iter(ds)
    s = next(it)
    assert isinstance(s, Sample)


def test_keys_subset(rellis_root):
    root, _ = rellis_root
    ds = Rellis3DDataset(root, keys=["lidar"])
    s = ds[0]
    assert "lidar" in s.data
    assert "labels" not in s.data


def test_invalid_key(rellis_root):
    root, _ = rellis_root
    with pytest.raises(KeyError):
        Rellis3DDataset(root, keys=["nonexistent"])


def test_out_of_range(rellis_root):
    root, total = rellis_root
    ds = Rellis3DDataset(root, keys=["lidar"])
    with pytest.raises(IndexError):
        ds[total]


def test_is_synchronous(rellis_root):
    root, _ = rellis_root
    ds = Rellis3DDataset(root, keys=["lidar"])
    assert ds.timestamps is None
    assert ds.is_synchronous is True


@pytest.fixture
def rellis_root_with_poses(tmp_path):
    n_frames = 3
    for seq in ["00000", "00001"]:
        bin_dir = tmp_path / "Rellis-3D" / seq / "os1_cloud_node_kitti_bin"
        lbl_dir = tmp_path / "Rellis-3D" / seq / "os1_cloud_node_semantickitti_label_id"
        bin_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        for i in range(n_frames):
            _make_bin(bin_dir / f"{i:06d}.bin")
            _make_label(lbl_dir / f"{i:06d}.label")
        _make_poses(tmp_path / "Rellis-3D" / seq / "poses.txt", n_frames)
    return tmp_path, n_frames * 2


def test_poses_shape(rellis_root_with_poses):
    root, total = rellis_root_with_poses
    ds = Rellis3DDataset(root, keys=["lidar", "poses"])
    assert len(ds) == total
    sample = ds[0]
    assert "poses" in sample.data
    assert sample.data["poses"].shape == (3, 4)


def test_poses_optional_absent(rellis_root):
    """poses is optional -- absent files should not appear in sample.data."""
    root, total = rellis_root
    ds = Rellis3DDataset(root, keys=["lidar", "labels", "poses"])
    assert len(ds) == total
    assert "poses" not in ds.keys


def test_mismatched_file_counts(tmp_path):
    """Init must fail if lidar and labels file counts differ."""
    bin_dir = tmp_path / "Rellis-3D" / "00000" / "os1_cloud_node_kitti_bin"
    lbl_dir = tmp_path / "Rellis-3D" / "00000" / "os1_cloud_node_semantickitti_label_id"
    bin_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    np.random.rand(60, 4).astype(np.float32).tofile(bin_dir / "000000.bin")
    np.random.rand(60, 4).astype(np.float32).tofile(bin_dir / "000001.bin")
    np.random.randint(0, 20, 60, dtype=np.int32).tofile(lbl_dir / "000000.label")
    with pytest.raises(ValueError):
        Rellis3DDataset(tmp_path, keys=["lidar", "labels"])


# ---------------------------------------------------------------------------
# Derived key tests
# ---------------------------------------------------------------------------

N_FRAMES_DERIVED = 3
N_ELEV = 32


def _write_apairo(root: Path, key: str, loader: str) -> None:
    import yaml

    config = {
        "version": 1,
        "channels": {
            key: {"kind": "preprocess", "loader": loader}
        },
    }
    d = root / ".apairo"
    d.mkdir(exist_ok=True)
    with open(d / "channels.yaml", "w") as f:
        yaml.dump(config, f)


@pytest.fixture
def rellis_root_derived(tmp_path):
    for seq in ["00000", "00001"]:
        bin_dir = tmp_path / "Rellis-3D" / seq / "os1_cloud_node_kitti_bin"
        bin_dir.mkdir(parents=True)
        elev_dir = tmp_path / "Rellis-3D" / seq / "elevation_map"
        elev_dir.mkdir(parents=True)
        for i in range(N_FRAMES_DERIVED):
            np.random.rand(N_POINTS, 4).astype(np.float32).tofile(
                bin_dir / f"{i:06d}.bin"
            )
            np.save(
                elev_dir / f"{i:06d}.npy", np.random.rand(N_ELEV).astype(np.float32)
            )
    _write_apairo(tmp_path, "elevation_map", "npys")
    return tmp_path


def test_derived_key_loaded_from_apairo(rellis_root_derived):
    ds = Rellis3DDataset(rellis_root_derived, keys=["lidar", "elevation_map"])
    assert len(ds) == N_FRAMES_DERIVED * 2
    sample = ds[0]
    assert "elevation_map" in sample.data
    assert isinstance(sample.data["elevation_map"], np.ndarray)


def test_derived_key_without_apairo_raises(tmp_path):
    bin_dir = tmp_path / "Rellis-3D" / "00000" / "os1_cloud_node_kitti_bin"
    bin_dir.mkdir(parents=True)
    elev_dir = tmp_path / "Rellis-3D" / "00000" / "elevation_map"
    elev_dir.mkdir(parents=True)
    for i in range(2):
        np.random.rand(N_POINTS, 4).astype(np.float32).tofile(bin_dir / f"{i:06d}.bin")
        np.save(elev_dir / f"{i:06d}.npy", np.random.rand(N_ELEV).astype(np.float32))
    with pytest.raises(KeyError):
        Rellis3DDataset(tmp_path, keys=["lidar", "elevation_map"])


def test_derived_key_missing_files_raises(tmp_path):
    bin_dir = tmp_path / "Rellis-3D" / "00000" / "os1_cloud_node_kitti_bin"
    bin_dir.mkdir(parents=True)
    for i in range(2):
        np.random.rand(N_POINTS, 4).astype(np.float32).tofile(bin_dir / f"{i:06d}.bin")
    _write_apairo(tmp_path, "elevation_map", "npys")
    with pytest.raises(FileNotFoundError):
        Rellis3DDataset(tmp_path, keys=["lidar", "elevation_map"])


# ---------------------------------------------------------------------------
# Split tests
# ---------------------------------------------------------------------------

N_SPLIT_FRAMES = 5
N_TRAIN = 3
N_VAL = 2


def _write_lst(path: Path, entries: list[tuple[str, str]]) -> None:
    lines = [
        f"{seq}/os1_cloud_node_kitti_bin/{stem}.bin"
        f" {seq}/os1_cloud_node_semantickitti_label_id/{stem}.label"
        for seq, stem in entries
    ]
    path.write_text("\n".join(lines))


@pytest.fixture
def rellis_root_with_splits(tmp_path):
    seqs = ["00000", "00001"]
    for seq in seqs:
        bin_dir = tmp_path / "Rellis-3D" / seq / "os1_cloud_node_kitti_bin"
        lbl_dir = tmp_path / "Rellis-3D" / seq / "os1_cloud_node_semantickitti_label_id"
        bin_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        for i in range(N_SPLIT_FRAMES):
            _make_bin(bin_dir / f"{i:06d}.bin")
            _make_label(lbl_dir / f"{i:06d}.label")

    # train: first N_TRAIN frames of each seq; val: remaining N_VAL
    train_entries = [(seq, f"{i:06d}") for seq in seqs for i in range(N_TRAIN)]
    val_entries = [
        (seq, f"{i:06d}") for seq in seqs for i in range(N_TRAIN, N_SPLIT_FRAMES)
    ]
    _write_lst(tmp_path / "pt_train.lst", train_entries)
    _write_lst(tmp_path / "pt_val.lst", val_entries)

    return tmp_path


def test_split_train_len(rellis_root_with_splits):
    ds = Rellis3DDataset(
        rellis_root_with_splits, keys=["lidar", "labels"], split="train"
    )
    assert len(ds) == N_TRAIN * 2  # 2 sequences


def test_split_val_len(rellis_root_with_splits):
    ds = Rellis3DDataset(rellis_root_with_splits, keys=["lidar", "labels"], split="val")
    assert len(ds) == N_VAL * 2


def test_split_and_sequence_ids(rellis_root_with_splits):
    ds = Rellis3DDataset(
        rellis_root_with_splits,
        keys=["lidar", "labels"],
        split="train",
        sequences=["00000"],
    )
    assert len(ds) == N_TRAIN


def test_splits_property(rellis_root_with_splits):
    ds = Rellis3DDataset(rellis_root_with_splits, keys=["lidar"])
    assert set(ds.splits) == {"train", "val", "test"}


def test_split_method(rellis_root_with_splits):
    ds = Rellis3DDataset(rellis_root_with_splits, keys=["lidar", "labels"])
    train_ds = ds.split("train")
    assert len(train_ds) == N_TRAIN * 2


def test_split_invalid(rellis_root_with_splits):
    with pytest.raises(ValueError, match="not declared in profile"):
        Rellis3DDataset(rellis_root_with_splits, keys=["lidar"], split="unknown")


def test_derived_only_without_source(rellis_root_derived):
    ds = Rellis3DDataset(rellis_root_derived, keys=["elevation_map"])
    assert len(ds) == N_FRAMES_DERIVED * 2
    s = ds[0]
    assert "elevation_map" in s.data
    assert "lidar" not in s.data
    assert isinstance(s.data["elevation_map"], np.ndarray)
