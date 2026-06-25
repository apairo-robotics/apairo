import pytest
import numpy as np
import yaml
from pathlib import Path
from apairo.core.profiled_dataset import ModalitySpec, _parse_layers, ProfiledDataset


def test_modality_spec_from_dict_basic():
    spec = ModalitySpec.from_dict(
        "lidar", {"ext": ".bin", "dtype": "float32", "reshape": [-1, 4]}
    )
    assert spec.ext == ".bin"
    assert spec.dtype == "float32"
    assert spec.reshape == [-1, 4]
    assert spec.mask is None
    assert spec.cast_dtype is None
    assert spec.optional is False
    assert spec.effective_subpath("lidar") == ["lidar"]


def test_modality_spec_ext_normalised():
    spec = ModalitySpec.from_dict("labels", {"ext": "label", "dtype": "int32"})
    assert spec.ext == ".label"


def test_modality_spec_with_all_fields():
    spec = ModalitySpec.from_dict(
        "labels",
        {
            "ext": ".label",
            "dtype": "int32",
            "mask": 65535,
            "cast_dtype": "int64",
            "subpath": ["camera", "left"],
            "optional": True,
        },
    )
    assert spec.mask == 65535
    assert spec.cast_dtype == "int64"
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
    # Mirrors real GOOSE: root/train/lidar/train/seq/file.bin (fixture omits first split -- glob is permissive)
    for seq in ["seq_a", "seq_b"]:
        (tmp_path / "lidar" / "train" / seq).mkdir(parents=True)
        (tmp_path / "labels" / "train" / seq).mkdir(parents=True)
        for i in range(3):
            _make_bin(tmp_path / "lidar" / "train" / seq / f"{i:06d}.bin")
            _make_label(tmp_path / "labels" / "train" / seq / f"{i:06d}.label")
    return tmp_path  # 2 seqs x 3 frames = 6 total


@pytest.fixture
def kitti_root(tmp_path):
    for seq in ["00", "01"]:
        (tmp_path / "sequences" / seq / "velodyne").mkdir(parents=True)
        (tmp_path / "sequences" / seq / "labels").mkdir(parents=True)
        for i in range(4):
            _make_bin(tmp_path / "sequences" / seq / "velodyne" / f"{i:06d}.bin")
            _make_label(tmp_path / "sequences" / seq / "labels" / f"{i:06d}.label")
    return tmp_path  # 2 seqs x 4 frames = 8 total


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
    assert ds._modality_idx == 0  # lidar/train/seq/file -> parts[0]="lidar"


def test_kitti_len(kitti_root):
    ds = _KittiDS(kitti_root, keys=["lidar", "labels"])
    assert len(ds) == 8


def test_kitti_modality_idx(kitti_root):
    ds = _KittiDS(kitti_root, keys=["lidar"])
    assert ds._modality_idx == 2  # sequences/00/velodyne/file -> parts[2]="velodyne"


def test_invalid_key_raises(goose_root):
    with pytest.raises(KeyError):
        _GooseDS(goose_root, keys=["nonexistent"])


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
    assert s.data["lidar"].dtype == np.float32
    assert s.data["labels"].dtype == np.int64  # int32 promoted via cast_dtype


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


class _RellisDS(ProfiledDataset):
    _profile = "rellis.yaml"


@pytest.fixture
def rellis_root(tmp_path):
    for seq in ["00000", "00001"]:
        d = tmp_path / "Rellis-3D" / seq / "os1_cloud_node_kitti_bin"
        d.mkdir(parents=True)
        for i in range(3):
            _make_bin(d / f"{i:06d}.bin")
    return tmp_path


def test_goose_derived_path_structure(goose_root):
    ds = _GooseDS(goose_root, keys=["lidar"])
    p = ds.derived_path(0, "trav_label", "npy")
    rel = p.relative_to(goose_root)
    assert rel.parts[0] == "trav_label"  # modality replaced at idx=0
    assert rel.parts[-1] == "000000.npy"


def test_kitti_derived_path_structure(kitti_root):
    ds = _KittiDS(kitti_root, keys=["lidar"])
    p = ds.derived_path(0, "trav_label", "npy")
    rel = p.relative_to(kitti_root)
    # sequences/00/velodyne/000000.bin -> sequences/00/trav_label/000000.npy
    assert rel.parts[0] == "sequences"
    assert rel.parts[2] == "trav_label"
    assert rel.parts[-1] == "000000.npy"


def test_rellis_derived_path_structure(rellis_root):
    ds = _RellisDS(rellis_root, keys=["lidar"])
    p = ds.derived_path(0, "trav_label", "npy")
    rel = p.relative_to(rellis_root)
    # Rellis-3D/00000/os1_cloud_node_kitti_bin/000000.bin -> Rellis-3D/00000/trav_label/000000.npy
    assert rel.parts[0] == "Rellis-3D"
    assert rel.parts[2] == "trav_label"
    assert rel.parts[-1] == "000000.npy"


def test_goose_seq_root(goose_root):
    ds = _GooseDS(goose_root, keys=["lidar"])
    first_file = ds._files["lidar"][0]
    seq = ds._seq_root(first_file)
    # GOOSE: train/lidar/train/seq_a/000000.bin -> _seq_depth=1 -> seq_root = first_file.parent
    assert seq == first_file.parent


def test_kitti_seq_root(kitti_root):
    ds = _KittiDS(kitti_root, keys=["lidar"])
    first_file = ds._files["lidar"][0]
    seq = ds._seq_root(first_file)
    # sequences/00/velodyne/000000.bin -> _seq_depth=2 -> seq_root = first_file.parent.parent
    assert seq == first_file.parent.parent


def test_bootstrap_config_uses_profile(goose_root):
    ds = _GooseDS(goose_root, keys=["lidar", "labels"])
    cfg = ds._bootstrap_config(goose_root)
    assert "lidar" in cfg["channels"]
    assert "labels" in cfg["channels"]
    assert cfg["channels"]["lidar"]["loader"] == "bin"
    assert cfg["channels"]["labels"]["loader"] == "bin"


def _write_apairo(directory: Path, key: str, loader: str) -> None:
    config = {
        "version": 1,
        "channels": {
            key: {"kind": "preprocess", "loader": loader}
        },
    }
    d = directory / ".apairo"
    d.mkdir(exist_ok=True)
    with open(d / "channels.yaml", "w") as f:
        yaml.dump(config, f)


@pytest.fixture
def goose_root_with_derived(goose_root):
    # .apairo registered at dataset root (GOOSE stores derived at root level)
    _write_apairo(goose_root, "elevation_map", "npys")
    for seq in ["seq_a", "seq_b"]:
        d = goose_root / "elevation_map" / "train" / seq
        d.mkdir(parents=True)
        for i in range(3):
            np.save(d / f"{i:06d}.npy", np.random.rand(32).astype(np.float32))
    return goose_root


def test_derived_key_loaded(goose_root_with_derived):
    ds = _GooseDS(goose_root_with_derived, keys=["lidar", "elevation_map"])
    assert len(ds) == 6
    s = ds[0]
    assert "elevation_map" in s.data
    assert isinstance(s.data["elevation_map"], np.ndarray)


def test_derived_only(goose_root_with_derived):
    ds = _GooseDS(goose_root_with_derived, keys=["elevation_map"])
    assert len(ds) == 6
    s = ds[0]
    assert "elevation_map" in s.data
    assert isinstance(s.data["elevation_map"], np.ndarray)


def test_derived_without_apairo_raises(goose_root):
    with pytest.raises(KeyError):
        _GooseDS(goose_root, keys=["lidar", "elevation_map"])


def test_derived_missing_files_raises(goose_root):
    _write_apairo(goose_root, "elevation_map", "npys")
    # No actual elevation_map files on disk
    with pytest.raises(FileNotFoundError):
        _GooseDS(goose_root, keys=["lidar", "elevation_map"])


# --- sequence_ids filter ---


def test_kitti_sequence_ids_filter(kitti_root):
    ds = _KittiDS(kitti_root, keys=["lidar", "labels"], sequences=["00"])
    assert len(ds) == 4  # only seq "00" (4 frames), not "01"
    assert ds.sequence_ids == ["00"]


def test_kitti_sequence_ids_filter_multiple(kitti_root):
    ds = _KittiDS(kitti_root, keys=["lidar", "labels"], sequences=["00", "01"])
    assert len(ds) == 8  # both sequences


def test_kitti_sequence_ids_filter_empty_result(kitti_root):
    with pytest.raises(FileNotFoundError):
        _KittiDS(kitti_root, keys=["lidar"], sequences=["99"])


def test_goose_sequence_ids_filter(goose_root):
    ds = _GooseDS(goose_root, keys=["lidar"], sequences=["seq_a"])
    assert len(ds) == 3  # only seq_a (3 frames)
    assert ds.sequence_ids == ["seq_a"]


def test_rellis_sequence_ids_filter(rellis_root):
    ds = _RellisDS(rellis_root, keys=["lidar"], sequences=["00000"])
    assert len(ds) == 3  # only seq "00000"
    assert ds.sequence_ids == ["00000"]


def test_sequence_ids_none_loads_all(kitti_root):
    ds = _KittiDS(kitti_root, keys=["lidar"], sequences=None)
    assert len(ds) == 8


def test_sequence_ids_param_deprecated_alias(kitti_root):
    """The old 'sequence_ids=' keyword still works, with a DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="sequences="):
        ds = _KittiDS(kitti_root, keys=["lidar"], sequence_ids=["00"])
    assert ds.sequence_ids == ["00"]  # property keeps its name


# --- describe ---


def test_describe_raw_present(rellis_root):
    ds = _RellisDS(rellis_root, keys=["lidar"])
    result = ds.describe()
    assert "lidar" in result["raw"]["present"]
    assert "lidar" not in result["raw"]["missing"]


def test_describe_raw_missing(rellis_root):
    # labels not on disk in this fixture
    ds = _RellisDS(rellis_root, keys=["lidar"])
    result = ds.describe()
    assert "labels" in result["raw"]["missing"]


def test_describe_with_sequence_id(rellis_root, capsys):
    ds = _RellisDS(rellis_root, keys=["lidar"])
    ds.describe("00000")
    out = capsys.readouterr().out
    assert "00000" in out


def test_describe_no_crash_without_apairo(kitti_root):
    ds = _KittiDS(kitti_root, keys=["lidar"])
    result = ds.describe()
    assert "raw" in result
    assert "preprocess" in result


def test_describe_structure(rellis_root):
    info = _RellisDS(rellis_root, keys=["lidar"]).describe()
    assert info["class"] == "_RellisDS"  # the actual class name
    assert info["sequences"] == ["00000", "00001"]
    # canonical channel name resolves to its real on-disk subdir (profile mapping)
    assert info["raw"]["channels"]["lidar"]["dir"] == "os1_cloud_node_kitti_bin"
    assert info["raw"]["channels"]["lidar"]["present"] is True
    assert info["layout"]["fixed"] == ["Rellis-3D"]


def test_inventory_matches_describe_without_instance(rellis_root):
    # inventory() is the path-based form: no constructor, no file discovery.
    inv = _RellisDS.inventory(rellis_root)
    ds_info = _RellisDS(rellis_root, keys=["lidar"]).describe()
    assert inv == ds_info


# ---------------------------------------------------------------------------
# Derived-from-derived
# ---------------------------------------------------------------------------


@pytest.fixture
def goose_root_chained(goose_root):
    """Chain: lidar (raw) -> elevation_map (preprocess) -> traversability (preprocess)."""
    config = {
        "version": 1,
        "channels": {
            "elevation_map": {
                "kind": "preprocess",
                "loader": "npys",
                "sources": ["lidar"],
            },
            "traversability": {
                "kind": "preprocess",
                "loader": "npys",
                "sources": ["elevation_map"],
            },
        },
    }
    apairo_dir = goose_root / ".apairo"
    apairo_dir.mkdir(exist_ok=True)
    with open(apairo_dir / "channels.yaml", "w") as f:
        yaml.dump(config, f)
    for seq in ["seq_a", "seq_b"]:
        for key in ["elevation_map", "traversability"]:
            d = goose_root / key / "train" / seq
            d.mkdir(parents=True)
            for i in range(3):
                np.save(d / f"{i:06d}.npy", np.random.rand(8).astype(np.float32))
    return goose_root


def test_derived_from_derived(goose_root_chained):
    ds = _GooseDS(goose_root_chained, keys=["traversability"])
    assert len(ds) == 6
    s = ds[0]
    assert "traversability" in s.data
    assert "elevation_map" not in s.data
    assert "lidar" not in s.data
    assert isinstance(s.data["traversability"], np.ndarray)


def test_derived_from_derived_with_intermediate(goose_root_chained):
    ds = _GooseDS(goose_root_chained, keys=["elevation_map", "traversability"])
    assert len(ds) == 6
    s = ds[0]
    assert "elevation_map" in s.data
    assert "traversability" in s.data


# --- frame_sequence_ids ---

def test_frame_sequence_ids_shape(kitti_root):
    ds = _KittiDS(kitti_root, keys=["lidar"])
    ids = ds.frame_sequence_ids
    assert len(ids) == len(ds)


def test_frame_sequence_ids_values(kitti_root):
    ds = _KittiDS(kitti_root, keys=["lidar"])
    ids = ds.frame_sequence_ids
    # kitti_root has seq "00" (4 frames) and "01" (4 frames)
    assert set(ids) == {"00", "01"}


def test_frame_sequence_ids_contiguous(kitti_root):
    ds = _KittiDS(kitti_root, keys=["lidar"])
    ids = ds.frame_sequence_ids
    # all frames of seq "00" come before all frames of seq "01"
    first_01 = np.where(ids == "01")[0][0]
    assert all(ids[:first_01] == "00")


def test_frame_sequence_ids_with_filter(kitti_root):
    """Split a filtered dataset by sequence without a second sweep."""
    ds = _KittiDS(kitti_root, keys=["lidar"])
    # keep first 3 frames of each sequence (all 8 frames pass a trivially true filter)
    view = ds.filter(lambda s: True)

    seq_ids = ds.frame_sequence_ids[view.indices]

    train_idx = np.where(seq_ids == "00")[0]
    val_idx   = np.where(seq_ids == "01")[0]

    ds_train = view.filter(train_idx)
    ds_val   = view.filter(val_idx)

    assert len(ds_train) == 4
    assert len(ds_val)   == 4
    assert len(ds_train) + len(ds_val) == len(view)


# ──────────────────────────── aliases on a profile ───────────────────────────
# Aliases were a RawDataset-only feature; ProfiledDataset ignored the `alias`
# field and raised KeyError on a request by alias. These pin the parity: a
# profiled dataset now honours aliases too, so channel names can be unified
# across heterogeneous datasets in one pipeline.

def test_profiled_alias_request_by_alias(goose_root):
    from apairo.core.config import set_alias

    _GooseDS(goose_root, keys=["lidar"])      # bootstraps .apairo/channels.yaml
    set_alias(goose_root, "lidar", "points")  # expose lidar as 'points'

    ds = _GooseDS(goose_root, keys=["points", "labels"])
    assert len(ds) == 6
    assert set(ds[0].data) == {"points", "labels"}   # exposed under the alias
    assert "lidar" not in ds[0].data
    assert ds.keys == ["points", "labels"]


def test_profiled_alias_request_by_real_name_still_resolves(goose_root):
    from apairo.core.config import set_alias

    _GooseDS(goose_root, keys=["lidar"])
    set_alias(goose_root, "lidar", "points")

    # Asking by the on-disk/profile name works, but the channel is exposed as the alias.
    ds = _GooseDS(goose_root, keys=["lidar"])
    assert set(ds[0].data) == {"points"}


def test_profiled_alias_default_keys_use_public_name(goose_root):
    from apairo.core.config import set_alias

    _GooseDS(goose_root, keys=["lidar"])
    set_alias(goose_root, "lidar", "points")

    ds = _GooseDS(goose_root)  # keys=None -> defaults exposed under the alias
    assert "points" in ds.keys and "lidar" not in ds.keys


def test_profiled_alias_survives_split(goose_root):
    from apairo.core.config import set_alias

    _GooseDS(goose_root, keys=["lidar", "labels"])
    set_alias(goose_root, "lidar", "points")

    ds = _GooseDS(goose_root, keys=["points", "labels"])
    ds_train = ds.split("train")  # round-trips keys (aliases) through the constructor
    assert set(ds_train[0].data) == {"points", "labels"}
