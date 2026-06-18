import pytest
import numpy as np
from pathlib import Path

from apairo.dataset.goose import Goose3DDataset
from apairo.core.sample import Sample

N_POINTS = 80


def _make_bin(path: Path, n: int = N_POINTS):
    np.random.rand(n, 4).astype(np.float32).tofile(path)


def _make_label(path: Path, n: int = N_POINTS):
    np.random.randint(0, 64, n, dtype=np.int32).tofile(path)


# Real GOOSE layout: <root>/split/lidar/split/seq/file.bin
#                    <root>/split/labels/split/seq/file.label
@pytest.fixture
def goose_root(tmp_path):
    n_frames = 5
    for seq in ["seq_a", "seq_b"]:
        lidar_dir = tmp_path / "lidar" / "train" / seq
        label_dir = tmp_path / "labels" / "train" / seq
        lidar_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for i in range(n_frames):
            _make_bin(lidar_dir / f"{i:06d}.bin")
            _make_label(label_dir / f"{i:06d}.label")
    return tmp_path, n_frames * 2  # 2 sequences x n_frames


def test_len(goose_root):
    root, total = goose_root
    ds = Goose3DDataset(root, keys=["lidar", "labels"])
    assert len(ds) == total


def test_getitem_returns_sample(goose_root):
    root, _ = goose_root
    ds = Goose3DDataset(root, keys=["lidar", "labels"])
    s = ds[0]
    assert isinstance(s, Sample)
    assert s.timestamp is None
    assert s.data["lidar"].shape == (N_POINTS, 4)
    assert s.data["labels"].shape == (N_POINTS,)
    assert s.data["lidar"].dtype == np.float32
    assert s.data["labels"].dtype == np.int64


def test_iter(goose_root):
    root, total = goose_root
    ds = Goose3DDataset(root, keys=["lidar", "labels"])
    assert len(list(ds)) == total


def test_next(goose_root):
    root, _ = goose_root
    ds = Goose3DDataset(root, keys=["lidar"])
    it = iter(ds)
    s = next(it)
    assert isinstance(s, Sample)


def test_keys_subset(goose_root):
    root, _ = goose_root
    ds = Goose3DDataset(root, keys=["lidar"])
    s = ds[0]
    assert "lidar" in s.data
    assert "labels" not in s.data


def test_invalid_key(goose_root):
    root, _ = goose_root
    with pytest.raises(KeyError):
        Goose3DDataset(root, keys=["nonexistent"])


def test_out_of_range(goose_root):
    root, total = goose_root
    ds = Goose3DDataset(root, keys=["lidar"])
    with pytest.raises(IndexError):
        ds[total]


def test_is_synchronous(goose_root):
    root, _ = goose_root
    ds = Goose3DDataset(root, keys=["lidar"])
    assert ds.timestamps is None
    assert ds.is_synchronous is True


# ---------------------------------------------------------------------------
# Derived key tests
# ---------------------------------------------------------------------------

N_FRAMES_DERIVED = 3
N_ELEV = 48


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
def goose_root_derived(tmp_path):
    # GOOSE layout: lidar/train/seq/file.bin
    # derived mirrors it:  elevation_map/train/seq/file.npy
    # .apairo at root (registration happens at dataset root)
    for seq in ["seq_a", "seq_b"]:
        lidar_dir = tmp_path / "lidar" / "train" / seq
        elev_dir = tmp_path / "elevation_map" / "train" / seq
        lidar_dir.mkdir(parents=True)
        elev_dir.mkdir(parents=True)
        for i in range(N_FRAMES_DERIVED):
            _make_bin(lidar_dir / f"{i:06d}.bin")
            np.save(
                elev_dir / f"{i:06d}.npy", np.random.rand(N_ELEV).astype(np.float32)
            )
    _write_apairo(tmp_path, "elevation_map", "npys")
    return tmp_path


def test_derived_key_loaded_from_apairo(goose_root_derived):
    ds = Goose3DDataset(goose_root_derived, keys=["lidar", "elevation_map"])
    assert len(ds) == N_FRAMES_DERIVED * 2
    sample = ds[0]
    assert "elevation_map" in sample.data
    assert isinstance(sample.data["elevation_map"], np.ndarray)


def test_derived_path_mirrors_modality_structure(goose_root_derived):
    ds = Goose3DDataset(goose_root_derived, keys=["lidar"])
    p = ds.derived_path(0, "trav_label", "npy")
    assert "trav_label" in p.parts
    assert p.suffix == ".npy"
    assert p.stem == "000000"
    # lidar component is replaced, rest of path is preserved
    lidar_ref = ds._files["lidar"][0].relative_to(ds.root_dir)
    derived_rel = p.relative_to(ds.root_dir)
    parts_ref = list(lidar_ref.parts)
    parts_der = list(derived_rel.parts)
    assert parts_der[ds._modality_idx] == "trav_label"
    assert parts_ref[1:-1] == parts_der[1:-1]  # sub-path after modality is identical


def test_derived_path_works_from_dataset_root(tmp_path):
    # root = GOOSE_3D/ with train/lidar/train/seq/ structure
    for seq in ["seq_a"]:
        lidar_dir = tmp_path / "train" / "lidar" / "train" / seq
        lidar_dir.mkdir(parents=True)
        for i in range(2):
            _make_bin(lidar_dir / f"{i:06d}.bin")
    ds = Goose3DDataset(tmp_path, keys=["lidar"])
    p = ds.derived_path(0, "trav_label", "npy")
    # Expect: tmp_path/train/trav_label/train/seq_a/000000.npy
    rel = p.relative_to(tmp_path)
    assert rel.parts[1] == "trav_label"  # modality replaced at idx=1
    assert rel.parts[0] == "train"  # split prefix preserved
    assert rel.parts[2] == "train"  # sub-split preserved


def test_derived_key_without_apairo_raises(tmp_path):
    lidar_dir = tmp_path / "lidar" / "train" / "seq_a"
    elev_dir = tmp_path / "elevation_map" / "train" / "seq_a"
    lidar_dir.mkdir(parents=True)
    elev_dir.mkdir(parents=True)
    for i in range(2):
        _make_bin(lidar_dir / f"{i:06d}.bin")
        np.save(elev_dir / f"{i:06d}.npy", np.zeros(N_ELEV, dtype=np.float32))
    with pytest.raises(KeyError):
        Goose3DDataset(tmp_path, keys=["lidar", "elevation_map"])


def test_derived_key_missing_files_raises(tmp_path):
    lidar_dir = tmp_path / "lidar" / "train" / "seq_a"
    lidar_dir.mkdir(parents=True)
    for i in range(2):
        _make_bin(lidar_dir / f"{i:06d}.bin")
    _write_apairo(tmp_path, "elevation_map", "npys")
    with pytest.raises(FileNotFoundError):
        Goose3DDataset(tmp_path, keys=["lidar", "elevation_map"])
