"""Tests for RawDataset -- the profile-free, channels.yaml-driven loader.

RawDataset is the asynchronous-layout member with *no* fixed channel set: the
channels come entirely from ``.apairo/channels.yaml`` (per sequence) and the
sequence order/name from an optional ``.apairo/dataset.yaml`` manifest -- the
layout produced by ``apairo-extractor``.  These tests fabricate that layout on
disk (npy / npys, plus a zarr channel to lock format-agnosticism) and exercise
both the single-sequence and the multi-sequence (root) paths.
"""
import numpy as np
import pytest
import yaml

from apairo.core.config import CONFIG_DIR, write_config
from apairo.core.sample import Sample
from apairo.dataset.kitti.dataset import _detect_loader
from apairo.dataset.raw import RawDataset


# ─────────────────────────────── helpers ─────────────────────────────────────

def _write_timestamps(channel_dir, ts):
    np.savetxt(channel_dir / "timestamps.txt", np.asarray(ts, dtype=float))


def _make_npys_channel(seq_dir, name, frames, ts):
    """A per-frame channel: one ``NNNNNN.npy`` per message (loader ``npys``)."""
    d = seq_dir / name
    d.mkdir(parents=True)
    for i, frame in enumerate(frames):
        np.save(d / f"{i:06d}.npy", frame)
    _write_timestamps(d, ts)


def _make_npy_channel(seq_dir, name, stacked, ts):
    """A buffered channel: one stacked ``.npy`` (loader ``npy``)."""
    d = seq_dir / name
    d.mkdir(parents=True)
    np.save(d / f"{name}.npy", stacked)
    _write_timestamps(d, ts)


def _write_channels(seq_dir, channels):
    write_config(
        seq_dir,
        {
            "version": 1,
            "channels": {
                k: {"loader": ldr, "kind": "raw"}
                for k, ldr in channels.items()
            },
        },
    )


def _write_manifest(root, name, sequences):
    d = root / CONFIG_DIR
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "dataset.yaml", "w") as f:
        yaml.dump({"version": 1, "name": name, "sequences": list(sequences)}, f)


def _make_sequence(seq_dir, n_lidar):
    """A sequence with a per-frame ``lidar`` (npys) and a buffered ``imu`` (npy)."""
    n_imu = n_lidar + 2
    _make_npys_channel(
        seq_dir,
        "lidar",
        [np.random.rand(4, 3) for _ in range(n_lidar)],
        np.linspace(0.0, 1.0, n_lidar),
    )
    _make_npy_channel(
        seq_dir,
        "imu",
        np.random.rand(n_imu, 6),
        np.linspace(0.0, 1.0, n_imu),
    )
    _write_channels(seq_dir, {"lidar": "npys", "imu": "npy"})
    return n_lidar + n_imu  # interleaved (asynchronous) timeline length


@pytest.fixture
def root_dataset(tmp_path):
    """A 2-sequence root with a manifest fixing a non-sorted order + a name."""
    root = tmp_path / "my_root"
    len_a = _make_sequence(root / "seq_a", n_lidar=3)
    len_b = _make_sequence(root / "seq_b", n_lidar=2)
    # Manifest order is deliberately the reverse of sorted discovery.
    _write_manifest(root, name="my_raw", sequences=["seq_b", "seq_a"])
    return root, len_a, len_b


# ─────────────────────────────── single sequence ─────────────────────────────

def test_single_sequence_loads_from_channels_yaml(tmp_path):
    seq = tmp_path / "seq_a"
    expected_len = _make_sequence(seq, n_lidar=3)

    ds = RawDataset(seq)

    assert ds.available == frozenset({"lidar", "imu"})
    assert set(ds.keys) == {"lidar", "imu"}
    assert len(ds) == expected_len  # 3 lidar + 5 imu, interleaved
    assert ds.name == "seq_a"
    assert isinstance(ds[0], Sample)


def test_single_sequence_keys_restrict(tmp_path):
    seq = tmp_path / "seq_a"
    _make_sequence(seq, n_lidar=3)

    ds = RawDataset(seq, keys=["lidar"])

    assert ds.keys == ["lidar"]
    assert len(ds) == 3


def test_single_sequence_is_not_root(tmp_path):
    seq = tmp_path / "seq_a"
    _make_sequence(seq, n_lidar=3)
    ds = RawDataset(seq)
    for attr in ("sequences", "sequence_ids"):
        with pytest.raises(AttributeError):
            getattr(ds, attr)


# ─────────────────────────────── root dataset ────────────────────────────────

def test_root_loads_all_sequences(root_dataset):
    root, len_a, len_b = root_dataset
    ds = RawDataset(root)

    assert len(ds.sequences) == 2
    assert len(ds) == len_a + len_b
    assert ds.available == frozenset({"lidar", "imu"})


def test_root_honours_manifest_name_and_order(root_dataset):
    root, _, _ = root_dataset
    ds = RawDataset(root)
    # Manifest declared name "my_raw" and order [seq_b, seq_a] (not sorted).
    assert ds.name == "my_raw"
    assert ds.sequence_ids == ["seq_b", "seq_a"]


def test_root_discovery_without_manifest(tmp_path):
    root = tmp_path / "no_manifest"
    _make_sequence(root / "seq_a", n_lidar=3)
    _make_sequence(root / "seq_b", n_lidar=2)
    ds = RawDataset(root)
    # No manifest -> sorted discovery, name falls back to the directory name.
    assert ds.sequence_ids == ["seq_a", "seq_b"]
    assert ds.name == "no_manifest"


def test_root_flat_indexing(root_dataset):
    root, len_a, len_b = root_dataset
    ds = RawDataset(root)
    assert isinstance(ds[0], Sample)
    assert isinstance(ds[len(ds) - 1], Sample)
    with pytest.raises(IndexError):
        ds[len(ds)]


def test_root_synchronize_concats_per_sequence(root_dataset):
    root, _, _ = root_dataset
    ds = RawDataset(root)
    synced = ds.synchronize(reference="lidar")
    # One synchronous frame per lidar message: 3 (seq_a) + 2 (seq_b).
    assert len(synced) == 5
    sample = synced[0]
    assert {"lidar", "imu"} <= set(sample.data)


def test_root_external_clock_rejected(root_dataset):
    root, _, _ = root_dataset
    ds = RawDataset(root)
    with pytest.raises(ValueError):
        ds.synchronize(reference=np.array([0.0, 0.1]))


# ─────────────────────────────── errors ──────────────────────────────────────

def test_neither_sequence_nor_root(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        RawDataset(empty)


# ─────────────────────────────── format-agnostic (zarr) ──────────────────────

def test_zarr_channel_is_format_agnostic(tmp_path):
    """An async channel stored as zarr loads identically -- the loader, taken
    from channels.yaml, is the only thing that changes (format is orthogonal)."""
    zarr = pytest.importorskip("zarr", reason="zarr not installed")

    seq = tmp_path / "seq_z"
    _make_npys_channel(
        seq, "lidar", [np.random.rand(4, 3) for _ in range(2)], [0.0, 0.1]
    )

    # The channel directory *is* the zarr store; timestamps.txt sits beside it.
    gps = seq / "gps"
    store = zarr.storage.LocalStore(str(gps))
    gps_data = np.random.rand(2, 3)
    arr = zarr.create(
        store=store, shape=gps_data.shape, dtype=gps_data.dtype,
        zarr_format=2, overwrite=True,
    )
    arr[:] = gps_data
    _write_timestamps(gps, [0.0, 0.1])

    _write_channels(seq, {"lidar": "npys", "gps": "zarr"})

    # Detection recognizes the zarr store directory.
    assert _detect_loader(gps) == "zarr"

    ds = RawDataset(seq, keys=["gps"])
    assert ds.available >= frozenset({"gps"})
    # Frame 0 of the gps channel is row 0 of the stored array.
    sample = ds[0]
    np.testing.assert_allclose(sample.data["gps"], gps_data[0])
