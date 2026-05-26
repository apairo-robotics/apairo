import pytest
import numpy as np
import torch
from pathlib import Path
from apairo.core.profiled_dataset import ModalitySpec, _parse_layers, LayerSpec, ProfiledDataset

def test_modality_spec_from_dict_basic():
    spec = ModalitySpec.from_dict("lidar", {"ext": ".bin", "dtype": "float32", "reshape": [-1, 4]})
    assert spec.ext == ".bin"
    assert spec.dtype == "float32"
    assert spec.reshape == [-1, 4]
    assert spec.mask is None
    assert spec.torch_dtype is None
    assert spec.optional is False
    assert spec.effective_subpath("lidar") == ["lidar"]

def test_modality_spec_ext_normalised():
    spec = ModalitySpec.from_dict("labels", {"ext": "label", "dtype": "int32"})
    assert spec.ext == ".label"

def test_modality_spec_with_all_fields():
    spec = ModalitySpec.from_dict("labels", {
        "ext": ".label", "dtype": "int32", "mask": 65535,
        "torch_dtype": "int64", "subpath": ["camera", "left"], "optional": True,
    })
    assert spec.mask == 65535
    assert spec.torch_dtype == "int64"
    assert spec.effective_subpath("labels") == ["camera", "left"]
    assert spec.optional is True

def test_parse_layers_goose():
    raw = [
        {"split": ["train", "val", "test"]},
        "modality",
        {"split": ["train", "val", "test"]},
        "sequence",
    ]
    layers = _parse_layers(raw)
    assert len(layers) == 4
    assert layers[0].type == "split"
    assert layers[0].value == ["train", "val", "test"]
    assert layers[1].type == "modality"
    assert layers[1].value is None
    assert layers[3].type == "sequence"

def test_parse_layers_kitti():
    raw = [
        {"fixed": "sequences"},
        "sequence",
        {"modality": {"lidar": "velodyne", "labels": "labels"}},
    ]
    layers = _parse_layers(raw)
    assert layers[0].type == "fixed"
    assert layers[0].value == "sequences"
    assert layers[2].type == "modality"
    assert layers[2].value == {"lidar": "velodyne", "labels": "labels"}


N_POINTS = 40

def _make_bin(path, n=N_POINTS):
    np.random.rand(n, 4).astype(np.float32).tofile(path)

def _make_label(path, n=N_POINTS):
    np.random.randint(0, 64, n, dtype=np.int32).tofile(path)


@pytest.fixture
def goose_root(tmp_path):
    # Mirrors real GOOSE: root/lidar/train/seq/file.bin
    for seq in ["seq_a", "seq_b"]:
        (tmp_path / "lidar" / "train" / seq).mkdir(parents=True)
        (tmp_path / "labels" / "train" / seq).mkdir(parents=True)
        for i in range(3):
            _make_bin(tmp_path / "lidar" / "train" / seq / f"{i:06d}.bin")
            _make_label(tmp_path / "labels" / "train" / seq / f"{i:06d}.label")
    return tmp_path  # 2 seqs × 3 frames = 6 total


@pytest.fixture
def kitti_root(tmp_path):
    for seq in ["00", "01"]:
        (tmp_path / "sequences" / seq / "velodyne").mkdir(parents=True)
        (tmp_path / "sequences" / seq / "labels").mkdir(parents=True)
        for i in range(4):
            _make_bin(tmp_path / "sequences" / seq / "velodyne" / f"{i:06d}.bin")
            _make_label(tmp_path / "sequences" / seq / "labels" / f"{i:06d}.label")
    return tmp_path  # 2 seqs × 4 frames = 8 total


class _GooseDS(ProfiledDataset):
    _profile = "goose.yaml"

class _KittiDS(ProfiledDataset):
    _profile = "semantic_kitti.yaml"


def test_goose_len(goose_root):
    ds = _GooseDS(goose_root, keys=["lidar", "labels"])
    assert len(ds) == 6

def test_goose_available_keys():
    assert _GooseDS.available_keys == frozenset({"lidar", "labels"})

def test_goose_modality_idx(goose_root):
    ds = _GooseDS(goose_root, keys=["lidar"])
    assert ds._modality_idx == 0  # lidar/train/seq/file → parts[0]="lidar"

def test_kitti_len(kitti_root):
    ds = _KittiDS(kitti_root, keys=["lidar", "labels"])
    assert len(ds) == 8

def test_kitti_modality_idx(kitti_root):
    ds = _KittiDS(kitti_root, keys=["lidar"])
    assert ds._modality_idx == 2  # sequences/00/velodyne/file → parts[2]="velodyne"

def test_invalid_key_raises(goose_root):
    with pytest.raises(KeyError):
        _GooseDS(goose_root, keys=["nonexistent"])

def test_mismatched_counts_raises(tmp_path):
    (tmp_path / "lidar" / "train" / "seq_a").mkdir(parents=True)
    (tmp_path / "labels" / "train" / "seq_a").mkdir(parents=True)
    _make_bin(tmp_path / "lidar" / "train" / "seq_a" / "000000.bin")
    _make_bin(tmp_path / "lidar" / "train" / "seq_a" / "000001.bin")
    _make_label(tmp_path / "labels" / "train" / "seq_a" / "000000.label")
    with pytest.raises(ValueError):
        _GooseDS(tmp_path, keys=["lidar", "labels"])

def test_missing_native_key_raises(tmp_path):
    (tmp_path / "lidar" / "train" / "seq_a").mkdir(parents=True)
    _make_bin(tmp_path / "lidar" / "train" / "seq_a" / "000000.bin")
    with pytest.raises(FileNotFoundError):
        _GooseDS(tmp_path, keys=["labels"])

def test_goose_split_filter(goose_root):
    ds = _GooseDS(goose_root, keys=["lidar"], split="train")
    assert len(ds) == 6  # all files are under "train"

def test_goose_split_filter_no_match(goose_root):
    with pytest.raises(FileNotFoundError):
        _GooseDS(goose_root, keys=["lidar"], split="val")

def test_goose_getitem_shapes(goose_root):
    ds = _GooseDS(goose_root, keys=["lidar", "labels"])
    s = ds[0]
    assert s.data["lidar"].shape == (N_POINTS, 4)
    assert s.data["labels"].shape == (N_POINTS,)
    assert s.data["lidar"].dtype == torch.float32
    assert s.data["labels"].dtype == torch.int64  # int32 promoted via torch_dtype

def test_goose_iter_count(goose_root):
    ds = _GooseDS(goose_root, keys=["lidar"])
    assert len(list(ds)) == 6

def test_goose_out_of_range(goose_root):
    ds = _GooseDS(goose_root, keys=["lidar"])
    with pytest.raises(IndexError):
        ds[6]

def test_kitti_label_mask(kitti_root):
    # Write labels with instance bits set in upper 16 bits
    lbl_path = sorted((kitti_root / "sequences" / "00" / "labels").glob("*.label"))[0]
    bin_path = sorted((kitti_root / "sequences" / "00" / "velodyne").glob("*.bin"))[0]
    np.array([0x00010001, 0x00020002], dtype=np.int32).tofile(lbl_path)
    np.random.rand(2, 4).astype(np.float32).tofile(bin_path)

    ds = _KittiDS(kitti_root, keys=["labels"])
    s = ds[0]
    assert s.data["labels"][0].item() == 0x0001
    assert s.data["labels"][1].item() == 0x0002

def test_is_synchronous(goose_root):
    ds = _GooseDS(goose_root, keys=["lidar"])
    assert ds.timestamps is None
    assert ds.is_synchronous is True
