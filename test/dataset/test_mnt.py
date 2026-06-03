"""Tests for MNTDataset -- the apairo wrapper for AMIAD MNT pipeline output.

The mock fixtures replicate the exact zarr v2 layout produced by
``src/_3_io/writer.py`` in the AMIAD_MNT_Dataset pipeline.

Data dimensions used throughout:
    N_FRAMES = 8    -- frames per mission
    N_POINTS = 64   -- LiDAR points per frame  (annotated: 5 cols)
    N_WPTS   = 10   -- waypoints per frame in traj_time/dist
    N_PAST   = 5    -- past-window size in trajectory.zarr
    H, W     = 12, 16 -- image spatial dims
"""

import io
import tarfile
import numpy as np
import pytest
from pathlib import Path

zarr = pytest.importorskip("zarr", reason="zarr not installed")
PIL_Image = pytest.importorskip("PIL.Image", reason="Pillow not installed")

from apairo.dataset.mnt.dataset import MNTDataset, _is_mission_dir
from apairo.core.sample import Sample

# ─────────────────────────────── constants ───────────────────────────────────

N_FRAMES = 8
N_POINTS = 64
N_WPTS   = 10
N_PAST   = 5
H, W     = 12, 16


# ─────────────────────────────── helpers ─────────────────────────────────────

def _make_zarr_v2(path: Path, data: np.ndarray) -> None:
    """Write a zarr v2 array (same flags as writer.py)."""
    store = zarr.storage.LocalStore(str(path))
    arr = zarr.create(
        store=store,
        shape=data.shape,
        dtype=data.dtype,
        zarr_format=2,
        overwrite=True,
    )
    arr[:] = data


def _make_images_tar(tar_path: Path, n: int) -> None:
    """Write a tar of n JPEG images named '{i:06d}.jpg'."""
    from PIL import Image as PImage

    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w") as tf:
        for i in range(n):
            img = PImage.fromarray(
                np.random.randint(0, 255, (H, W, 3), dtype=np.uint8), mode="RGB"
            )
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            buf.seek(0)
            info = tarfile.TarInfo(name=f"{i:06d}.jpg")
            info.size = len(buf.getvalue())
            buf.seek(0)
            tf.addfile(info, buf)


def make_mission(base: Path, name: str = "mission_001") -> Path:
    """Build a complete mock mission directory matching the MNT writer output."""
    m = base / name
    m.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    # ── images.tar ────────────────────────────────────────────────────────
    _make_images_tar(m / "images.tar", N_FRAMES)

    # ── points.zarr  (N, P, 5) annotated cloud ───────────────────────────
    _make_zarr_v2(
        m / "points.zarr",
        rng.random((N_FRAMES, N_POINTS, 5), dtype=np.float32),
    )

    # ── trajectory.zarr ───────────────────────────────────────────────────
    traj = m / "trajectory.zarr"
    traj.mkdir()
    positions = rng.random((N_FRAMES, 2), dtype=np.float32)
    yaws      = rng.random((N_FRAMES,),   dtype=np.float32)
    ts        = np.linspace(0.0, 7.0, N_FRAMES, dtype=np.float64)

    _make_zarr_v2(traj / "positions.zarr",         positions)
    _make_zarr_v2(traj / "yaws.zarr",              yaws)
    _make_zarr_v2(traj / "timestamps.zarr",        ts)
    _make_zarr_v2(traj / "positions_past.zarr",    rng.random((N_FRAMES, N_PAST, 2), dtype=np.float32))
    _make_zarr_v2(traj / "yaws_past.zarr",         rng.random((N_FRAMES, N_PAST),    dtype=np.float32))
    _make_zarr_v2(traj / "timestamps_past.zarr",   rng.random((N_FRAMES, N_PAST),    dtype=np.float64))

    # ── traj_time.zarr ────────────────────────────────────────────────────
    tt = m / "traj_time.zarr"
    tt.mkdir()
    _make_zarr_v2(tt / "positions.zarr", rng.random((N_FRAMES, N_WPTS, 2), dtype=np.float32))
    _make_zarr_v2(tt / "yaws.zarr",      rng.random((N_FRAMES, N_WPTS),    dtype=np.float32))

    # ── traj_dist.zarr ────────────────────────────────────────────────────
    td = m / "traj_dist.zarr"
    td.mkdir()
    _make_zarr_v2(td / "positions.zarr", rng.random((N_FRAMES, N_WPTS, 2), dtype=np.float32))
    _make_zarr_v2(td / "yaws.zarr",      rng.random((N_FRAMES, N_WPTS),    dtype=np.float32))

    return m


# ─────────────────────────────── fixtures ────────────────────────────────────

@pytest.fixture
def mission_dir(tmp_path):
    return make_mission(tmp_path)


@pytest.fixture
def dataset_root(tmp_path):
    """Two missions under a common root."""
    make_mission(tmp_path, "mission_001")
    make_mission(tmp_path, "mission_002")
    return tmp_path, N_FRAMES * 2


# ═════════════════════════════════════════════════════════════════════════════
# Detection helpers
# ═════════════════════════════════════════════════════════════════════════════

def test_is_mission_dir_true(mission_dir):
    assert _is_mission_dir(mission_dir)


def test_is_mission_dir_false_on_root(tmp_path):
    make_mission(tmp_path, "mission_001")
    assert not _is_mission_dir(tmp_path)


def test_is_mission_dir_minimal(tmp_path):
    # Presence of just trajectory.zarr is enough
    (tmp_path / "trajectory.zarr").mkdir()
    assert _is_mission_dir(tmp_path)


# ═════════════════════════════════════════════════════════════════════════════
# Mission-level: basic loading
# ═════════════════════════════════════════════════════════════════════════════

def test_len_mission(mission_dir):
    ds = MNTDataset(mission_dir, keys=["position"])
    assert len(ds) == N_FRAMES


def test_getitem_returns_sample(mission_dir):
    ds = MNTDataset(mission_dir, keys=["position", "yaw", "timestamp"])
    s = ds[0]
    assert isinstance(s, Sample)
    assert s.timestamp is None          # synchronous dataset
    assert set(s.data.keys()) == {"position", "yaw", "timestamp"}


def test_position_shape_and_dtype(mission_dir):
    ds = MNTDataset(mission_dir, keys=["position"])
    pos = ds[0].data["position"]
    assert pos.shape == (2,)
    assert pos.dtype == np.float32


def test_yaw_scalar(mission_dir):
    ds = MNTDataset(mission_dir, keys=["yaw"])
    yaw = ds[0].data["yaw"]
    assert yaw.shape == ()
    assert yaw.dtype == np.float32


def test_timestamp_scalar(mission_dir):
    ds = MNTDataset(mission_dir, keys=["timestamp"])
    ts = ds[0].data["timestamp"]
    assert ts.shape == ()
    assert ts.dtype == np.float64


def test_points_shape(mission_dir):
    ds = MNTDataset(mission_dir, keys=["points"])
    pts = ds[0].data["points"]
    assert pts.shape == (N_POINTS, 5)   # annotated: x,y,z,ground_flag,trav_flag
    assert pts.dtype == np.float32


def test_image_shape(mission_dir):
    ds = MNTDataset(mission_dir, keys=["image"])
    img = ds[0].data["image"]
    assert img.ndim == 3
    assert img.shape[2] == 3
    assert img.dtype == np.uint8


def test_image_last_frame(mission_dir):
    ds = MNTDataset(mission_dir, keys=["image"])
    img = ds[N_FRAMES - 1].data["image"]
    assert img.shape == (H, W, 3)


def test_waypoints_time_shape(mission_dir):
    ds = MNTDataset(mission_dir, keys=["waypoints_time"])
    wpt = ds[0].data["waypoints_time"]
    assert wpt.shape == (N_WPTS, 2)
    assert wpt.dtype == np.float32


def test_yaw_waypoints_time_shape(mission_dir):
    ds = MNTDataset(mission_dir, keys=["yaw_waypoints_time"])
    yw = ds[0].data["yaw_waypoints_time"]
    assert yw.shape == (N_WPTS,)
    assert yw.dtype == np.float32


def test_waypoints_dist_shape(mission_dir):
    ds = MNTDataset(mission_dir, keys=["waypoints_dist"])
    wpt = ds[0].data["waypoints_dist"]
    assert wpt.shape == (N_WPTS, 2)


def test_position_past_shape(mission_dir):
    ds = MNTDataset(mission_dir, keys=["position_past"])
    pp = ds[0].data["position_past"]
    assert pp.shape == (N_PAST, 2)
    assert pp.dtype == np.float32


def test_yaw_past_shape(mission_dir):
    ds = MNTDataset(mission_dir, keys=["yaw_past"])
    yp = ds[0].data["yaw_past"]
    assert yp.shape == (N_PAST,)


def test_timestamp_past_shape(mission_dir):
    ds = MNTDataset(mission_dir, keys=["timestamp_past"])
    tp = ds[0].data["timestamp_past"]
    assert tp.shape == (N_PAST,)
    assert tp.dtype == np.float64


# ═════════════════════════════════════════════════════════════════════════════
# Mission-level: default keys (auto-discover all present channels)
# ═════════════════════════════════════════════════════════════════════════════

def test_default_keys_all_channels_present(mission_dir):
    ds = MNTDataset(mission_dir)
    expected = {
        "image", "points",
        "position", "yaw", "timestamp",
        "position_past", "yaw_past", "timestamp_past",
        "waypoints_time", "yaw_waypoints_time",
        "waypoints_dist", "yaw_waypoints_dist",
    }
    assert set(ds.keys) == expected


def test_default_keys_partial(tmp_path):
    """Only trajectory.zarr present -- only those channels loaded."""
    m = tmp_path / "m"
    m.mkdir()
    traj = m / "trajectory.zarr"
    traj.mkdir()
    rng = np.random.default_rng(0)
    _make_zarr_v2(traj / "positions.zarr", rng.random((4, 2), dtype=np.float32))
    _make_zarr_v2(traj / "yaws.zarr",      rng.random((4,),   dtype=np.float32))
    _make_zarr_v2(traj / "timestamps.zarr", np.arange(4, dtype=np.float64))

    ds = MNTDataset(m)
    assert "position" in ds.keys
    assert "yaw"      in ds.keys
    assert "image"    not in ds.keys


# ═════════════════════════════════════════════════════════════════════════════
# Mission-level: iteration
# ═════════════════════════════════════════════════════════════════════════════

def test_iter_all_frames(mission_dir):
    ds = MNTDataset(mission_dir, keys=["position"])
    samples = list(ds)
    assert len(samples) == N_FRAMES
    assert all(isinstance(s, Sample) for s in samples)


def test_iter_positions_increase(mission_dir):
    """Timestamps should increase monotonically with frame index."""
    ds = MNTDataset(mission_dir, keys=["timestamp"])
    ts = [s.data["timestamp"].item() for s in ds]
    assert ts == sorted(ts)


def test_next_raises_stop_iteration(mission_dir):
    ds = MNTDataset(mission_dir, keys=["yaw"])
    it = iter(ds)
    for _ in range(N_FRAMES):
        next(it)
    with pytest.raises(StopIteration):
        next(it)


# ═════════════════════════════════════════════════════════════════════════════
# Mission-level: indexing edge cases
# ═════════════════════════════════════════════════════════════════════════════

def test_index_zero(mission_dir):
    ds = MNTDataset(mission_dir, keys=["position"])
    assert isinstance(ds[0], Sample)


def test_index_last(mission_dir):
    ds = MNTDataset(mission_dir, keys=["position"])
    assert isinstance(ds[N_FRAMES - 1], Sample)


def test_index_out_of_range(mission_dir):
    ds = MNTDataset(mission_dir, keys=["position"])
    with pytest.raises(IndexError):
        ds[N_FRAMES]


def test_index_negative_out_of_range(mission_dir):
    ds = MNTDataset(mission_dir, keys=["position"])
    with pytest.raises(IndexError):
        ds[-1]


# ═════════════════════════════════════════════════════════════════════════════
# Mission-level: keys
# ═════════════════════════════════════════════════════════════════════════════

def test_keys_subset(mission_dir):
    ds = MNTDataset(mission_dir, keys=["image", "position"])
    s = ds[0]
    assert set(s.data.keys()) == {"image", "position"}


def test_is_synchronous(mission_dir):
    ds = MNTDataset(mission_dir, keys=["position"])
    assert ds.timestamps is None
    assert ds.is_synchronous


def test_available_keys_complete(mission_dir):
    ds = MNTDataset(mission_dir)
    assert MNTDataset.available_keys >= {"image", "points", "position", "yaw",
                                          "timestamp", "waypoints_time", "waypoints_dist"}


# ═════════════════════════════════════════════════════════════════════════════
# Mission-level: sequence view
# ═════════════════════════════════════════════════════════════════════════════

def test_mission_ids_single(mission_dir):
    ds = MNTDataset(mission_dir, keys=["position"])
    assert ds.mission_ids == [mission_dir.name]


def test_sequences_single(mission_dir):
    ds = MNTDataset(mission_dir, keys=["position"])
    seqs = ds.sequences()
    assert len(seqs) == 1
    assert len(seqs[0]) == N_FRAMES


def test_sequence_by_name(mission_dir):
    ds = MNTDataset(mission_dir, keys=["position"])
    seq = ds.sequence(mission_dir.name)
    assert len(seq) == N_FRAMES
    assert isinstance(seq[0], Sample)


def test_sequence_unknown_name(mission_dir):
    ds = MNTDataset(mission_dir, keys=["position"])
    with pytest.raises(KeyError):
        ds.sequence("nonexistent")


# ═════════════════════════════════════════════════════════════════════════════
# Root-level (multi-mission)
# ═════════════════════════════════════════════════════════════════════════════

def test_root_len(dataset_root):
    root, total = dataset_root
    ds = MNTDataset(root, keys=["position"])
    assert len(ds) == total


def test_root_iter_count(dataset_root):
    root, total = dataset_root
    ds = MNTDataset(root, keys=["position"])
    assert len(list(ds)) == total


def test_root_mission_ids(dataset_root):
    root, _ = dataset_root
    ds = MNTDataset(root, keys=["position"])
    assert set(ds.mission_ids) == {"mission_001", "mission_002"}


def test_root_sequences(dataset_root):
    root, _ = dataset_root
    ds = MNTDataset(root, keys=["position"])
    seqs = ds.sequences()
    assert len(seqs) == 2
    assert all(len(s) == N_FRAMES for s in seqs)


def test_root_sequence_by_name(dataset_root):
    root, _ = dataset_root
    ds = MNTDataset(root, keys=["position"])
    seq = ds.sequence("mission_001")
    assert len(seq) == N_FRAMES


def test_root_getitem_crosses_boundary(dataset_root):
    """Frame at index N_FRAMES should come from the second mission."""
    root, _ = dataset_root
    ds = MNTDataset(root, keys=["timestamp"])
    s_first  = ds[0].data["timestamp"].item()
    s_second = ds[N_FRAMES].data["timestamp"].item()
    # Both missions have the same timestamps; the important thing is that
    # index N_FRAMES is accessible (second mission, frame 0).
    assert isinstance(s_second, float)
    assert abs(s_second - s_first) < 1e-9  # same synthetic data


def test_root_getitem_out_of_range(dataset_root):
    root, total = dataset_root
    ds = MNTDataset(root, keys=["position"])
    with pytest.raises(IndexError):
        ds[total]


def test_root_no_missions_raises(tmp_path):
    (tmp_path / "not_a_mission").mkdir()
    with pytest.raises(FileNotFoundError):
        MNTDataset(tmp_path, keys=["position"])


# ═════════════════════════════════════════════════════════════════════════════
# Preprocessing integration
# ═════════════════════════════════════════════════════════════════════════════

class _ConstantFramePrep:
    """Minimal FramePreprocessor stub (no ABC, just duck-typed for runner)."""
    output_key    = "const_label"
    output_loader = "npys"
    input_keys    = ["position"]
    timestamps_from = "position"
    sources       = ["position"]

    def process(self, sample: Sample) -> np.ndarray:
        return np.array([1], dtype=np.uint8)


def test_derived_path_layout(mission_dir):
    ds = MNTDataset(mission_dir, keys=["position"])
    p = ds.derived_path(0, "trav_label", "npy")
    assert p.parent.name == "trav_label"
    assert p.name == "000000.npy"
    assert p.is_relative_to(mission_dir)


def test_derived_path_index_format(mission_dir):
    ds = MNTDataset(mission_dir, keys=["position"])
    p = ds.derived_path(7, "foo", "npy")
    assert p.name == "000007.npy"


def test_run_preprocess_writes_files(mission_dir):
    from apairo.core.preprocessor import FramePreprocessor

    class TravLabel(FramePreprocessor):
        output_key      = "trav_label"
        output_loader   = "npys"
        input_keys      = ["position"]
        timestamps_from = "position"
        sources         = ["position"]

        def process(self, sample: Sample) -> np.ndarray:
            return np.array([1], dtype=np.uint8)

    MNTDataset.run_preprocess(TravLabel(), mission_dir)

    # All N_FRAMES files should exist
    label_dir = mission_dir / "trav_label"
    files = sorted(label_dir.glob("*.npy"))
    assert len(files) == N_FRAMES
    assert files[0].name == "000000.npy"
    assert files[-1].name == f"{N_FRAMES - 1:06d}.npy"


def test_run_preprocess_registers_channel(mission_dir):
    from apairo.core.preprocessor import FramePreprocessor
    from apairo.core.config import config_exists, read_config

    class TravLabel(FramePreprocessor):
        output_key      = "trav_label"
        output_loader   = "npys"
        input_keys      = ["position"]
        timestamps_from = "position"
        sources         = ["position"]

        def process(self, sample: Sample) -> np.ndarray:
            return np.array([1], dtype=np.uint8)

    MNTDataset.run_preprocess(TravLabel(), mission_dir)
    assert config_exists(mission_dir)
    cfg = read_config(mission_dir)
    assert "trav_label" in cfg["channels"]
    assert cfg["channels"]["trav_label"]["kind"] == "preprocess"


def test_run_preprocess_no_overwrite_raises(mission_dir):
    from apairo.core.preprocessor import FramePreprocessor

    class TravLabel(FramePreprocessor):
        output_key      = "trav_label"
        output_loader   = "npys"
        input_keys      = ["position"]
        timestamps_from = "position"
        sources         = ["position"]

        def process(self, sample: Sample) -> np.ndarray:
            return np.array([1], dtype=np.uint8)

    MNTDataset.run_preprocess(TravLabel(), mission_dir)
    with pytest.raises(FileExistsError):
        MNTDataset.run_preprocess(TravLabel(), mission_dir, overwrite=False)


def test_load_preprocessed_channel(mission_dir):
    """After run_preprocess, the derived key is loadable from MNTDataset."""
    from apairo.core.preprocessor import FramePreprocessor

    class TravLabel(FramePreprocessor):
        output_key      = "trav_label"
        output_loader   = "npys"
        input_keys      = ["position"]
        timestamps_from = "position"
        sources         = ["position"]

        def process(self, sample: Sample) -> np.ndarray:
            return np.array([1], dtype=np.uint8)

    MNTDataset.run_preprocess(TravLabel(), mission_dir)
    ds = MNTDataset(mission_dir, keys=["position", "trav_label"])
    s = ds[0]
    assert "trav_label" in s.data
    assert s.data["trav_label"].item() == 1


# ═════════════════════════════════════════════════════════════════════════════
# describe()
# ═════════════════════════════════════════════════════════════════════════════

def test_describe_mission(mission_dir, capsys):
    result = MNTDataset.describe(mission_dir)
    out = capsys.readouterr().out
    assert "MNTDataset" in out
    assert "image" in result["raw"]["present"]
    assert "points" in result["raw"]["present"]
    assert "position" in result["raw"]["present"]


def test_describe_root(dataset_root, capsys):
    root, _ = dataset_root
    result = MNTDataset.describe(root)
    out = capsys.readouterr().out
    assert "mission_001" in out
    assert "mission_002" in out
    assert "mission_001" in result
    assert "mission_002" in result


# ═════════════════════════════════════════════════════════════════════════════
# Loader unit tests
# ═════════════════════════════════════════════════════════════════════════════

def test_zarr_loader_len(tmp_path):
    from apairo.loader.zarr_loader import ZarrLoader
    p = tmp_path / "arr.zarr"
    _make_zarr_v2(p, np.arange(12, dtype=np.float32).reshape(6, 2))
    loader = ZarrLoader(p)
    assert len(loader) == 6


def test_zarr_loader_getitem(tmp_path):
    from apairo.loader.zarr_loader import ZarrLoader
    data = np.arange(12, dtype=np.float32).reshape(6, 2)
    p = tmp_path / "arr.zarr"
    _make_zarr_v2(p, data)
    loader = ZarrLoader(p)
    np.testing.assert_array_equal(loader[2], data[2])


def test_zarr_loader_1d(tmp_path):
    from apairo.loader.zarr_loader import ZarrLoader
    data = np.linspace(0, 1, 8, dtype=np.float32)
    p = tmp_path / "scalar.zarr"
    _make_zarr_v2(p, data)
    loader = ZarrLoader(p)
    assert loader[3].shape == ()
    np.testing.assert_almost_equal(float(loader[3]), data[3])


def test_tar_image_loader_len(mission_dir):
    from apairo.loader.tar_loader import TarImageLoader
    loader = TarImageLoader(mission_dir / "images.tar", N_FRAMES)
    assert len(loader) == N_FRAMES


def test_tar_image_loader_shape(mission_dir):
    from apairo.loader.tar_loader import TarImageLoader
    loader = TarImageLoader(mission_dir / "images.tar", N_FRAMES)
    img = loader[0]
    assert img.shape == (H, W, 3)
    assert img.dtype == np.uint8


def test_tar_image_loader_last_frame(mission_dir):
    from apairo.loader.tar_loader import TarImageLoader
    loader = TarImageLoader(mission_dir / "images.tar", N_FRAMES)
    img = loader[N_FRAMES - 1]
    assert img.shape == (H, W, 3)


def test_tar_image_loader_index_caches(mission_dir):
    from apairo.loader.tar_loader import TarImageLoader
    loader = TarImageLoader(mission_dir / "images.tar", N_FRAMES)
    _ = loader[0]                 # builds index
    _ = loader[1]                 # uses cached index
    assert loader._index is not None
    assert len(loader._index) == N_FRAMES


def test_tar_image_loader_missing_raises(mission_dir):
    from apairo.loader.tar_loader import TarImageLoader
    loader = TarImageLoader(mission_dir / "images.tar", N_FRAMES + 1)
    with pytest.raises(FileNotFoundError):
        loader[N_FRAMES]          # index N_FRAMES was never written


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════

def test_public_api_export():
    import apairo
    assert hasattr(apairo, "MNTDataset")
    assert apairo.MNTDataset is MNTDataset
