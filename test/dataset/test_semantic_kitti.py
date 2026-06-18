import pytest
import numpy as np
from pathlib import Path

from apairo.dataset.semantic_kitti import SemanticKittiDataset
from apairo.core.sample import Sample

N_POINTS = 50


def _make_bin(path: Path, n: int = N_POINTS):
    np.random.rand(n, 4).astype(np.float32).tofile(path)


def _make_label(path: Path, n: int = N_POINTS):
    np.random.randint(0, 20, n, dtype=np.int32).tofile(path)


@pytest.fixture
def kitti_root(tmp_path):
    n_frames = 4
    for seq in ["00", "01"]:
        vel = tmp_path / "sequences" / seq / "velodyne"
        lbl = tmp_path / "sequences" / seq / "labels"
        vel.mkdir(parents=True)
        lbl.mkdir(parents=True)
        for i in range(n_frames):
            _make_bin(vel / f"{i:06d}.bin")
            _make_label(lbl / f"{i:06d}.label")
    return tmp_path, n_frames * 2  # 2 sequences x n_frames


def test_len(kitti_root):
    root, total = kitti_root
    ds = SemanticKittiDataset(root, keys=["lidar", "labels"])
    assert len(ds) == total


def test_getitem_returns_sample(kitti_root):
    root, _ = kitti_root
    ds = SemanticKittiDataset(root, keys=["lidar", "labels"])
    s = ds[0]
    assert isinstance(s, Sample)
    assert s.timestamp is None
    assert s.data["lidar"].shape == (N_POINTS, 4)
    assert s.data["labels"].shape == (N_POINTS,)
    assert s.data["lidar"].dtype == np.float32
    assert s.data["labels"].dtype == np.int64


def test_iter(kitti_root):
    root, total = kitti_root
    ds = SemanticKittiDataset(root, keys=["lidar", "labels"])
    assert len(list(ds)) == total


def test_next(kitti_root):
    root, total = kitti_root
    ds = SemanticKittiDataset(root, keys=["lidar"])
    it = iter(ds)
    s = next(it)
    assert isinstance(s, Sample)


def test_keys_subset(kitti_root):
    root, _ = kitti_root
    ds = SemanticKittiDataset(root, keys=["lidar"])
    s = ds[0]
    assert "lidar" in s.data
    assert "labels" not in s.data


def test_invalid_key(kitti_root):
    root, _ = kitti_root
    with pytest.raises(KeyError):
        SemanticKittiDataset(root, keys=["nonexistent"])


def test_out_of_range(kitti_root):
    root, total = kitti_root
    ds = SemanticKittiDataset(root, keys=["lidar"])
    with pytest.raises(IndexError):
        ds[total]


def test_is_synchronous(kitti_root):
    root, _ = kitti_root
    ds = SemanticKittiDataset(root, keys=["lidar"])
    assert ds.timestamps is None
    assert ds.is_synchronous is True


def test_label_lower_16_bits(kitti_root):
    """SemanticKITTI labels encode instance in upper bits -- only lower 16 bits are semantic."""
    root, _ = kitti_root
    # Write a label file where upper bits are set
    path = sorted((root / "sequences" / "00" / "labels").glob("*.label"))[0]
    labels_with_instance = np.array([0x00010001, 0x00020002], dtype=np.int32)
    labels_with_instance.tofile(path)
    # Also rewrite the matching bin file with correct point count
    bin_path = sorted((root / "sequences" / "00" / "velodyne").glob("*.bin"))[0]
    np.random.rand(2, 4).astype(np.float32).tofile(bin_path)

    ds = SemanticKittiDataset(root, keys=["labels"])
    s = ds[0]
    assert s.data["labels"][0].item() == 0x0001
    assert s.data["labels"][1].item() == 0x0002


def test_mismatched_file_counts(tmp_path):
    """Init must fail if lidar and labels file counts differ."""
    vel = tmp_path / "sequences" / "00" / "velodyne"
    lbl = tmp_path / "sequences" / "00" / "labels"
    vel.mkdir(parents=True)
    lbl.mkdir(parents=True)
    np.random.rand(50, 4).astype(np.float32).tofile(vel / "000000.bin")
    np.random.rand(50, 4).astype(np.float32).tofile(vel / "000001.bin")
    np.random.randint(0, 20, 50, dtype=np.int32).tofile(lbl / "000000.label")
    # only one label file vs two lidar files
    with pytest.raises(ValueError):
        SemanticKittiDataset(tmp_path, keys=["lidar", "labels"])


# ---------------------------------------------------------------------------
# Derived key tests
# ---------------------------------------------------------------------------

N_FRAMES_DERIVED = 3
N_ELEV = 64


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
def kitti_root_derived(tmp_path):
    for seq in ["00", "01"]:
        vel = tmp_path / "sequences" / seq / "velodyne"
        vel.mkdir(parents=True)
        elev = tmp_path / "sequences" / seq / "elevation_map"
        elev.mkdir(parents=True)
        for i in range(N_FRAMES_DERIVED):
            np.random.rand(N_POINTS, 4).astype(np.float32).tofile(vel / f"{i:06d}.bin")
            np.save(elev / f"{i:06d}.npy", np.random.rand(N_ELEV).astype(np.float32))
    _write_apairo(tmp_path, "elevation_map", "npys")
    return tmp_path


def test_derived_key_loaded_from_apairo(kitti_root_derived):
    ds = SemanticKittiDataset(kitti_root_derived, keys=["lidar", "elevation_map"])
    assert len(ds) == N_FRAMES_DERIVED * 2
    sample = ds[0]
    assert "elevation_map" in sample.data
    assert isinstance(sample.data["elevation_map"], np.ndarray)


def test_derived_key_without_apairo_raises(tmp_path):
    vel = tmp_path / "sequences" / "00" / "velodyne"
    vel.mkdir(parents=True)
    elev = tmp_path / "sequences" / "00" / "elevation_map"
    elev.mkdir(parents=True)
    for i in range(2):
        np.random.rand(N_POINTS, 4).astype(np.float32).tofile(vel / f"{i:06d}.bin")
        np.random.rand(N_ELEV).astype(np.float32).tofile(elev / f"{i:06d}.npy")
    with pytest.raises(KeyError):
        SemanticKittiDataset(tmp_path, keys=["lidar", "elevation_map"])


def test_derived_key_missing_files_raises(tmp_path):
    vel = tmp_path / "sequences" / "00" / "velodyne"
    vel.mkdir(parents=True)
    for i in range(2):
        np.random.rand(N_POINTS, 4).astype(np.float32).tofile(vel / f"{i:06d}.bin")
    _write_apairo(tmp_path, "elevation_map", "npys")
    with pytest.raises(FileNotFoundError):
        SemanticKittiDataset(tmp_path, keys=["lidar", "elevation_map"])
