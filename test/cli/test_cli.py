"""Tests for the ``apairo`` CLI (init + status)."""
import json

import numpy as np
import pytest

from apairo.cli import main
from apairo.core.config import config_exists, read_config


def _make_seq(seq_dir, n_lidar):
    """A sequence with a per-frame ``lidar`` (npys) and a buffered ``imu`` (npy)."""
    (seq_dir / "lidar").mkdir(parents=True)
    for i in range(n_lidar):
        np.save(seq_dir / "lidar" / f"{i:06d}.npy", np.random.rand(4, 3))
    np.savetxt(seq_dir / "lidar" / "timestamps.txt", np.linspace(0, 1, n_lidar))
    (seq_dir / "imu").mkdir()
    np.save(seq_dir / "imu" / "imu.npy", np.random.rand(n_lidar + 2, 6))
    np.savetxt(seq_dir / "imu" / "timestamps.txt", np.linspace(0, 1, n_lidar + 2))


@pytest.fixture
def raw_root(tmp_path):
    root = tmp_path / "my_root"
    _make_seq(root / "seq_a", 3)  # 3 lidar + 5 imu = 8 events
    _make_seq(root / "seq_b", 2)  # 2 lidar + 4 imu = 6 events
    return root


def _run(argv) -> int:
    with pytest.raises(SystemExit) as exc:
        main(argv)
    return exc.value.code


# ── init ─────────────────────────────────────────────────────────────────────

def test_init_root_writes_sidecars(raw_root):
    assert _run(["init", str(raw_root), "--name", "ds"]) == 0
    assert (raw_root / ".apairo" / "dataset.yaml").is_file()
    for s in ("seq_a", "seq_b"):
        assert config_exists(raw_root / s)
        ch = read_config(raw_root / s)["channels"]
        assert ch["lidar"]["loader"] == "npys"
        assert ch["imu"]["loader"] == "npy"


def test_init_single_sequence(tmp_path):
    seq = tmp_path / "seq"
    _make_seq(seq, 3)
    assert _run(["init", str(seq)]) == 0
    assert config_exists(seq)
    assert not (seq / ".apairo" / "dataset.yaml").exists()  # sequence, not root


def test_init_is_idempotent(raw_root):
    assert _run(["init", str(raw_root)]) == 0
    assert _run(["init", str(raw_root)]) == 0  # second run must not error


def test_init_default_merge_picks_up_new_channel(raw_root):
    _run(["init", str(raw_root)])
    extra = raw_root / "seq_a" / "extra"
    extra.mkdir()
    np.save(extra / "extra.npy", np.random.rand(3, 2))
    np.savetxt(extra / "timestamps.txt", np.linspace(0, 1, 3))
    _run(["init", str(raw_root)])  # default = merge
    assert "extra" in read_config(raw_root / "seq_a")["channels"]


def test_init_not_a_directory(tmp_path):
    assert _run(["init", str(tmp_path / "missing")]) == 2


# ── status ───────────────────────────────────────────────────────────────────

def test_status_before_init_lists_untracked(raw_root, capsys):
    code = _run(["status", str(raw_root)])
    out = capsys.readouterr().out
    assert code == 0
    assert "untracked" in out
    assert "seq_a/lidar" in out


def test_status_after_init_json(raw_root, capsys):
    _run(["init", str(raw_root), "--name", "ds"])
    capsys.readouterr()  # discard init output
    _run(["status", str(raw_root), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["name"] == "ds"
    assert data["kind"] == "root"
    assert data["sequences"] == ["seq_a", "seq_b"]
    assert data["raw"] == {"lidar": "npys", "imu": "npy"}
    assert data["events"] == 14  # (3+5) + (2+4)
    assert data["untracked"] == []
    assert data["issues"] == []


def test_status_single_sequence_json(raw_root, capsys):
    _run(["init", str(raw_root)])
    capsys.readouterr()
    _run(["status", str(raw_root / "seq_a"), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["kind"] == "sequence"
    assert data["events"] == 8


def test_status_not_a_dataset(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _run(["status", str(empty)]) == 1


def test_status_sequence_per_channel_json(raw_root, capsys):
    _run(["init", str(raw_root)])
    capsys.readouterr()
    _run(["status", str(raw_root / "seq_a"), "--json"])
    ch = json.loads(capsys.readouterr().out)["channels"]
    assert ch["lidar"]["frames"] == 3
    assert ch["lidar"]["loader"] == "npys"
    assert ch["lidar"]["shape"] == [4, 3]          # per-frame shape (mmap header)
    assert ch["lidar"]["dtype"].startswith("float")
    assert ch["lidar"]["rate_hz"] == pytest.approx(2.0)   # (3-1)/(1-0)
    assert ch["imu"]["frames"] == 5
    assert ch["imu"]["loader"] == "npy"
    assert ch["imu"]["shape"] == [6]               # stacked (5, 6) -> per-frame (6,)
    assert ch["imu"]["rate_hz"] == pytest.approx(4.0)


def test_status_sequence_table_text(raw_root, capsys):
    _run(["init", str(raw_root)])
    capsys.readouterr()
    _run(["status", str(raw_root / "seq_a")])
    out = capsys.readouterr().out
    assert "rate" in out and "shape" in out
    assert "2.0 Hz" in out       # lidar rate
    assert "(4, 3)" in out       # lidar shape


def test_status_span_is_relative_to_earliest(tmp_path, capsys):
    seq = tmp_path / "seq"
    base = 1_779_893_201.0  # epoch-style absolute timestamps
    (seq / "a").mkdir(parents=True)
    np.save(seq / "a" / "a.npy", np.zeros((3, 2)))
    np.savetxt(seq / "a" / "timestamps.txt", base + np.array([0.0, 1.0, 2.0]))
    (seq / "b").mkdir()
    np.save(seq / "b" / "b.npy", np.zeros((3, 2)))
    np.savetxt(seq / "b" / "timestamps.txt", base + np.array([0.5, 1.5, 2.5]))
    _run(["init", str(seq)])
    capsys.readouterr()

    _run(["status", str(seq)])
    out = capsys.readouterr().out
    assert f"start       {base:.2f}s" in out   # absolute reference shown once
    assert "0.00–2.00s" in out                 # channel a, relative
    assert "0.50–2.50s" in out                 # channel b, relative offset preserved

    _run(["status", str(seq), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["start"] == base
    # JSON keeps absolute spans (ground truth) -- relative is for display only.
    assert data["channels"]["a"]["span"] == [base, base + 2.0]


def test_status_shows_channel_frame(raw_root, capsys):
    from apairo.core.config import register_raw_channel

    _run(["init", str(raw_root)])
    register_raw_channel(raw_root / "seq_a", "lidar", "npys", frame="lidar_link")
    capsys.readouterr()

    _run(["status", str(raw_root / "seq_a"), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["channels"]["lidar"]["frame"] == "lidar_link"
    assert data["channels"]["imu"]["frame"] is None  # not declared

    capsys.readouterr()
    _run(["status", str(raw_root / "seq_a")])
    out = capsys.readouterr().out
    assert "frame" in out and "lidar_link" in out  # column appears only when declared


def test_status_untracked_channel_detail(raw_root, capsys):
    _run(["init", str(raw_root)])
    # drop a new channel on disk without registering it
    seg = raw_root / "seq_a" / "segmentation"
    seg.mkdir()
    for i in range(3):
        np.save(seg / f"{i:06d}.npy", np.zeros((2, 2), dtype="uint8"))
    np.savetxt(seg / "timestamps.txt", np.linspace(0, 1, 3))
    capsys.readouterr()
    _run(["status", str(raw_root / "seq_a"), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert "segmentation" in data["untracked"]
    assert data["untracked"]["segmentation"]["kind"] == "untracked"
    assert data["untracked"]["segmentation"]["frames"] == 3
