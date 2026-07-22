"""Tests for ``AbstractDataset.export`` and the ``apairo export`` CLI.

v1: structural subset of the asynchronous RawDataset family -- whole sequences x
whole channels copied to a new self-contained root.
"""

from __future__ import annotations

import numpy as np
import pytest

from apairo.cli import main
from apairo.core.config import (
    config_exists,
    read_config,
    read_manifest,
    register_static_transform,
    verify_config,
)
from apairo.core.preprocessor import FramePreprocessor
from apairo.core.sample import Sample
from apairo.dataset.raw import RawDataset


def _make_seq(seq_dir, n_lidar, *, intensity=False):
    """A sequence: per-frame lidar (npys), buffered imu (npy), optional intensity."""
    (seq_dir / "lidar").mkdir(parents=True)
    for i in range(n_lidar):
        np.save(seq_dir / "lidar" / f"{i:06d}.npy", np.random.rand(4, 3).astype("f4"))
        if intensity:
            np.save(
                seq_dir / "lidar" / f"{i:06d}_intensity.npy",
                np.random.rand(4).astype("f4"),
            )
    np.savetxt(seq_dir / "lidar" / "timestamps.txt", np.linspace(0, 1, n_lidar))
    (seq_dir / "imu").mkdir()
    np.save(seq_dir / "imu" / "imu.npy", np.random.rand(n_lidar + 2, 6).astype("f4"))
    np.savetxt(seq_dir / "imu" / "timestamps.txt", np.linspace(0, 1, n_lidar + 2))


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "src"
    _make_seq(r / "seq_a", 3, intensity=True)
    _make_seq(r / "seq_b", 2, intensity=True)
    return r


class _ZMean(FramePreprocessor):
    output_key = "z_mean"
    output_loader = "npys"
    input_keys = ["lidar"]
    timestamps_from = "lidar"
    sources = ["lidar"]

    def __call__(self, sample: Sample) -> np.ndarray:
        return np.array([sample.data["lidar"][:, 2].mean()], dtype=np.float32)


def _run(argv) -> int:
    with pytest.raises(SystemExit) as exc:
        main(argv)
    return exc.value.code


# ── channel + sequence subset ─────────────────────────────────────────────────


def test_export_channel_subset(root, tmp_path):
    dest = tmp_path / "out"
    src = RawDataset(root, keys=["lidar", "imu"])
    out = src.export(dest)

    assert out == dest
    for seq in ("seq_a", "seq_b"):
        assert set(read_config(dest / seq)["channels"]) == {"lidar", "imu"}
        assert verify_config(dest / seq) == []  # lidar_intensity is not declared
    assert read_manifest(dest)["sequences"] == ["seq_a", "seq_b"]

    reloaded = RawDataset(dest, keys=["lidar", "imu"])
    assert len(reloaded) == len(src)
    np.testing.assert_array_equal(
        RawDataset(dest, keys=["imu"])[0].data["imu"],
        RawDataset(root, keys=["imu"])[0].data["imu"],
    )


def test_export_sequence_subset(root, tmp_path):
    dest = tmp_path / "out"
    RawDataset(root).filter_sequences(["seq_a"]).export(dest)

    assert config_exists(dest / "seq_a")
    assert not (dest / "seq_b").exists()
    assert read_manifest(dest)["sequences"] == ["seq_a"]
    assert set(RawDataset(dest).sequence_ids) == {"seq_a"}


def test_export_single_sequence_becomes_root(root, tmp_path):
    dest = tmp_path / "out"
    RawDataset(root / "seq_a", keys=["lidar"]).export(dest)

    # A single-sequence source exports as a one-sequence root.
    assert read_manifest(dest)["sequences"] == ["seq_a"]
    assert len(RawDataset(dest, keys=["lidar"])) == 3


# ── suffixed sub-channels ─────────────────────────────────────────────────────


def test_export_suffix_with_base(root, tmp_path):
    dest = tmp_path / "out"
    RawDataset(root, keys=["lidar", "lidar_intensity"]).filter_sequences(
        ["seq_a"]
    ).export(dest)

    ch = read_config(dest / "seq_a")["channels"]
    assert {"lidar", "lidar_intensity"} <= set(ch)
    assert ch["lidar_intensity"]["suffix"] == "intensity"
    out = RawDataset(dest, keys=["lidar_intensity"])  # _check_suffix_coverage passes
    assert len(out) == 3


def test_export_suffix_without_base(root, tmp_path):
    dest = tmp_path / "out"
    RawDataset(root, keys=["lidar_intensity"]).filter_sequences(["seq_a"]).export(dest)

    ch = read_config(dest / "seq_a")["channels"]
    assert set(ch) == {"lidar_intensity"}
    # The shared clock travelled with the directory, so the subset is loadable.
    assert verify_config(dest / "seq_a") == []
    assert len(RawDataset(dest, keys=["lidar_intensity"])) == 3


# ── derived channels ──────────────────────────────────────────────────────────


def test_export_derived_channel_is_self_contained(tmp_path):
    root = tmp_path / "src"
    _make_seq(root / "seq_a", 4)
    _make_seq(root / "seq_b", 3)
    RawDataset(root, keys=["lidar"]).run_preprocess(_ZMean())

    dest = tmp_path / "out"
    src = RawDataset(root, keys=["z_mean"])
    src.export(dest)

    ch = read_config(dest / "seq_a")["channels"]["z_mean"]
    assert "timestamps_from" not in ch  # source lidar not exported -> normalized away
    assert "sources" not in ch
    assert (dest / "seq_a" / "z_mean" / "timestamps.txt").exists()
    assert verify_config(dest / "seq_a") == []
    np.testing.assert_array_equal(
        RawDataset(dest, keys=["z_mean"])[0].data["z_mean"], src[0].data["z_mean"]
    )


def test_export_derived_keeps_provenance_when_source_travels(tmp_path):
    root = tmp_path / "src"
    _make_seq(root / "seq_a", 4)
    RawDataset(root / "seq_a", keys=["lidar"]).run_preprocess(_ZMean())

    dest = tmp_path / "out"
    RawDataset(root, keys=["lidar", "z_mean"]).export(dest)

    ch = read_config(dest / "seq_a")["channels"]["z_mean"]
    assert ch["timestamps_from"] == "lidar"  # source exported too -> kept
    assert ch["sources"] == ["lidar"]
    assert verify_config(dest / "seq_a") == []


# ── calibration + third-party sidecars ────────────────────────────────────────


def test_export_copies_calibration_drops_foreign_sidecars(root, tmp_path):
    RawDataset(root)  # bootstrap .apairo
    register_static_transform(root / "seq_a", "lidar", "base", np.eye(4))
    (root / "seq_a" / ".apairo" / "metadata.yaml").write_text("extractor: v1\n")

    dest = tmp_path / "out"
    RawDataset(root, keys=["lidar"]).filter_sequences(["seq_a"]).export(dest)

    assert (dest / "seq_a" / ".apairo" / "calibration.yaml").is_file()
    assert not (dest / "seq_a" / ".apairo" / "metadata.yaml").exists()


# ── hardlink ──────────────────────────────────────────────────────────────────


def test_export_link_shares_inodes(root, tmp_path):
    dest = tmp_path / "out"
    RawDataset(root, keys=["lidar"]).filter_sequences(["seq_a"]).export(dest, link=True)

    src_file = root / "seq_a" / "lidar" / "000000.npy"
    dst_file = dest / "seq_a" / "lidar" / "000000.npy"
    assert src_file.stat().st_ino == dst_file.stat().st_ino


# ── rejections + collisions ───────────────────────────────────────────────────


def test_export_rejects_synchronized_view(root, tmp_path):
    with pytest.raises(ValueError, match="materializing export|asynchronous"):
        RawDataset(root).synchronize(reference="lidar").export(tmp_path / "out")


def test_export_rejects_frame_filter(root, tmp_path):
    ds = RawDataset(root, keys=["lidar"])
    with pytest.raises(ValueError, match="frame-filtered"):
        ds.filter(np.arange(0, len(ds), 2)).export(tmp_path / "out")


def test_export_refuses_nonempty_dest(root, tmp_path):
    dest = tmp_path / "out"
    RawDataset(root, keys=["lidar"]).export(dest)
    with pytest.raises(FileExistsError):
        RawDataset(root, keys=["lidar"]).export(dest)
    RawDataset(root, keys=["imu"]).export(dest, overwrite=True)  # ok
    assert set(read_config(dest / "seq_a")["channels"]) == {"imu"}


def test_export_overwrite_replaces_stale_sequences(root, tmp_path):
    # overwrite must REPLACE, not merge: a seq_a-only re-export over a seq_a+seq_b
    # export must not leave seq_b behind to re-enter the regenerated manifest.
    dest = tmp_path / "out"
    RawDataset(root, keys=["lidar", "imu"]).export(dest)
    assert read_manifest(dest)["sequences"] == ["seq_a", "seq_b"]
    RawDataset(root, keys=["lidar", "imu"]).filter_sequences(["seq_a"]).export(
        dest, overwrite=True
    )
    assert read_manifest(dest)["sequences"] == ["seq_a"]  # was [seq_a, seq_b]
    assert not (dest / "seq_b").exists()


def test_export_keeps_channel_selected_by_its_real_name(root, tmp_path):
    # A channel opened by its REAL name while it carries an alias must not be
    # silently dropped from the export.
    from apairo.core.config import set_alias

    RawDataset(root)  # bootstrap .apairo per sequence
    for seq in ("seq_a", "seq_b"):
        set_alias(root / seq, "lidar", "points")  # alias the lidar channel
    dest = tmp_path / "out"
    RawDataset(root, keys=["lidar", "imu"]).export(dest)  # opened by real name
    assert set(read_config(dest / "seq_a")["channels"]) == {"lidar", "imu"}


# ── SequencePreprocessor on an async root respects sequence boundaries (H12) ───


def test_sequence_preprocessor_on_async_root_runs_per_sequence(root):
    from apairo.core.preprocessor import SequencePreprocessor

    class _SeqPos(SequencePreprocessor):
        # each frame's index within its own sequence -- resets to 0 at every seam
        output_key = "seq_pos"
        output_loader = "npys"
        input_keys = ["lidar"]
        timestamps_from = "lidar"
        sources = ["lidar"]

        def __call__(self, frames):
            return np.arange(len(list(frames)), dtype=np.int64)

    RawDataset(root, keys=["lidar"]).run_preprocess(_SeqPos())
    # each sequence gets its own output (seq_a=3 frames, seq_b=2), not one file
    assert len(list((root / "seq_a" / "seq_pos").glob("*.npy"))) == 3
    assert len(list((root / "seq_b" / "seq_pos").glob("*.npy"))) == 2
    ds = RawDataset(root, keys=["seq_pos"])
    pos = [int(ds[i].data["seq_pos"]) for i in range(len(ds))]
    assert pos == [0, 1, 2, 0, 1]  # resets at the seq_a->seq_b boundary (not 0..4)


# ── CLI ───────────────────────────────────────────────────────────────────────


def test_export_cli(root, tmp_path):
    dest = tmp_path / "out"
    code = _run(
        [
            "export",
            str(root),
            str(dest),
            "--keys",
            "lidar",
            "imu",
            "--sequences",
            "seq_a",
        ]
    )
    assert code == 0
    assert read_manifest(dest)["sequences"] == ["seq_a"]
    assert set(read_config(dest / "seq_a")["channels"]) == {"lidar", "imu"}


def test_export_cli_collision_returns_1(root, tmp_path):
    dest = tmp_path / "out"
    assert _run(["export", str(root), str(dest), "--keys", "lidar"]) == 0
    assert _run(["export", str(root), str(dest), "--keys", "lidar"]) == 1
