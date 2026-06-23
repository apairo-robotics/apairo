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


# ── profiled datasets (init --as <Class>) ─────────────────────────────────────


@pytest.fixture
def rellis_root(tmp_path):
    """A RELLIS-3D layout: <root>/Rellis-3D/<seq>/os1_cloud_node_kitti_bin/*.bin."""
    root = tmp_path / "rellis"
    for seq in ("00000", "00001"):
        d = root / "Rellis-3D" / seq / "os1_cloud_node_kitti_bin"
        d.mkdir(parents=True)
        for i in range(3):
            np.random.rand(40, 4).astype("float32").tofile(d / f"{i:06d}.bin")
    return root


def test_init_profiled_writes_manifest_class(rellis_root):
    import yaml

    assert _run(["init", str(rellis_root), "--as", "Rellis3DDataset"]) == 0
    manifest = yaml.safe_load((rellis_root / ".apairo" / "dataset.yaml").read_text())
    assert manifest["class"] == "Rellis3DDataset"  # identity persisted for status


def test_status_profiled_dispatches_through_profile(rellis_root, capsys):
    _run(["init", str(rellis_root), "--as", "Rellis3DDataset"])
    capsys.readouterr()
    _run(["status", str(rellis_root), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["class"] == "Rellis3DDataset"        # recognized, not generic
    assert data["kind"] == "root"
    assert data["sequences"] == ["00000", "00001"]   # sequences are visible
    assert data["raw"] == {"lidar": "bin"}           # canonical names, real loader
    assert data["events"] == 6                        # 2 seqs x 3 frames
    # no generic false-positives ("directory not found", "unknown loader txt_rows");
    # only the genuinely-missing required channel is reported.
    assert data["issues"] == [
        "raw channel 'labels' declared in Rellis3DDataset profile but not found on disk"
    ]


def test_status_profiled_text_header(rellis_root, capsys):
    _run(["init", str(rellis_root), "--as", "Rellis3DDataset"])
    capsys.readouterr()
    _run(["status", str(rellis_root)])
    out = capsys.readouterr().out
    assert "Rellis3DDataset - rellis" in out  # header names the dataset class
    assert "00000, 00001" in out


def test_status_profiled_sequence_by_id(rellis_root, capsys):
    # -s <id> drills into one sequence from the root, resolving canonical names
    # to their nested dirs -- impossible via `status Rellis-3D/00000`.
    _run(["init", str(rellis_root), "--as", "Rellis3DDataset"])
    capsys.readouterr()
    _run(["status", str(rellis_root), "-s", "00000", "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["kind"] == "sequence"
    assert data["class"] == "Rellis3DDataset"
    assert data["name"] == "00000"
    assert data["channels"]["lidar"]["loader"] == "bin"
    assert data["channels"]["lidar"]["frames"] == 3   # per-frame census of the resolved dir
    assert data["events"] == 3


def test_status_profiled_sequence_text_header(rellis_root, capsys):
    _run(["init", str(rellis_root), "--as", "Rellis3DDataset"])
    capsys.readouterr()
    _run(["status", str(rellis_root), "-s", "00001"])
    out = capsys.readouterr().out
    assert "Rellis3DDataset - 00001   (sequence)" in out
    assert "lidar" in out and "frames" in out


def test_status_sequence_unknown_id_errors(rellis_root, capsys):
    _run(["init", str(rellis_root), "--as", "Rellis3DDataset"])
    capsys.readouterr()
    assert _run(["status", str(rellis_root), "-s", "99999"]) == 1
    err = capsys.readouterr().err
    assert "not found" in err
    assert "00000" in err and "00001" in err  # available sequences listed


def test_status_generic_sequence_by_id(raw_root, capsys):
    # -s works uniformly on a generic root: resolves <root>/<id>.
    _run(["init", str(raw_root)])
    capsys.readouterr()
    _run(["status", str(raw_root), "-s", "seq_a", "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["kind"] == "sequence"
    assert data["name"] == "seq_a"
    assert data["events"] == 8   # same as `status raw_root/seq_a`


# ── ecosystem dispatch (apairo <tool>) ────────────────────────────────────────

class _FakeEntryPoint:
    def __init__(self, fn):
        self._fn = fn

    def load(self):
        return self._fn


def test_dispatches_to_ecosystem_plugin(monkeypatch):
    import apairo.cli as cli

    captured = {}
    monkeypatch.setattr(
        cli, "_discover_plugins",
        lambda: {"extractor": _FakeEntryPoint(lambda argv: captured.setdefault("argv", argv) or 0)},
    )
    with pytest.raises(SystemExit) as exc:
        cli.main(["extractor", "-i", "bags", "-o", "out"])
    assert captured["argv"] == ["-i", "bags", "-o", "out"]  # rest handed to the plugin
    assert exc.value.code == 0


def test_builtins_win_over_plugins(monkeypatch, raw_root):
    import apairo.cli as cli

    monkeypatch.setattr(
        cli, "_discover_plugins",
        lambda: {"extractor": _FakeEntryPoint(lambda argv: 1)},
    )
    # 'init' is a built-in -> handled by apairo, not dispatched as a plugin
    assert _run(["init", str(raw_root)]) == 0


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
    assert "0.00-2.00s" in out                 # channel a, relative
    assert "0.50-2.50s" in out                 # channel b, relative offset preserved

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


def test_status_shows_transform_edge(raw_root, capsys):
    from apairo.core.config import register_raw_channel

    _run(["init", str(raw_root)])
    edge = raw_root / "seq_a" / "odom__base_link"
    edge.mkdir()
    np.save(edge / "odom__base_link.npy", np.zeros((4, 7)))
    np.savetxt(edge / "timestamps.txt", np.linspace(0, 1, 4))
    register_raw_channel(
        raw_root / "seq_a", "odom__base_link", "npy",
        transform={"parent": "odom", "child": "base_link"},
    )
    capsys.readouterr()

    _run(["status", str(raw_root / "seq_a"), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["channels"]["odom__base_link"]["transform"] == {
        "parent": "odom", "child": "base_link",
    }

    # tf is hidden by default -- the transform channel and its edge are not shown
    capsys.readouterr()
    _run(["status", str(raw_root / "seq_a")])
    out = capsys.readouterr().out
    assert "odom->base_link" not in out
    assert "--show-tf" in out

    # --show-tf reveals the transform edge
    capsys.readouterr()
    _run(["status", str(raw_root / "seq_a"), "--show-tf"])
    assert "odom->base_link" in capsys.readouterr().out


def test_calibration_roundtrip_and_status(raw_root, capsys):
    from apairo.core.config import read_calibration, register_static_transform
    from apairo.dataset.raw import RawDataset

    _run(["init", str(raw_root)])
    M = np.eye(4)
    M[0, 3], M[2, 3] = 1.0, 0.5
    register_static_transform(raw_root / "seq_a", "base_link", "lidar", M)

    # read_calibration returns the 4x4
    calib = read_calibration(raw_root / "seq_a")
    np.testing.assert_allclose(calib["base_link_to_lidar"], M)

    # RawDataset.calibration property (sequence + root merge)
    assert "base_link_to_lidar" in RawDataset(raw_root / "seq_a").calibration
    np.testing.assert_allclose(
        RawDataset(raw_root).calibration["base_link_to_lidar"], M
    )

    # status surfaces it
    capsys.readouterr()
    _run(["status", str(raw_root / "seq_a"), "--json"])
    assert json.loads(capsys.readouterr().out)["calibration"] == ["base_link_to_lidar"]
    # calibration (static tf) is hidden by default, shown with --show-tf
    capsys.readouterr()
    _run(["status", str(raw_root / "seq_a")])
    assert "calibration" not in capsys.readouterr().out
    capsys.readouterr()
    _run(["status", str(raw_root / "seq_a"), "--show-tf"])
    assert "calibration" in capsys.readouterr().out


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


# ── channel aliases (apairo alias) ────────────────────────────────────────────

def test_alias_command_is_root_aware(raw_root):
    from apairo.core.config import read_config

    _run(["init", str(raw_root)])
    assert _run(["alias", "lidar", "points", "--path", str(raw_root)]) == 0
    # applied to every sequence holding the channel
    for s in ("seq_a", "seq_b"):
        assert read_config(raw_root / s)["channels"]["lidar"]["alias"] == "points"


def test_alias_command_remove(raw_root):
    from apairo.core.config import read_config

    _run(["init", str(raw_root)])
    _run(["alias", "lidar", "points", "--path", str(raw_root)])
    assert _run(["alias", "lidar", "--remove", "--path", str(raw_root)]) == 0
    assert "alias" not in read_config(raw_root / "seq_a")["channels"]["lidar"]


def test_alias_command_unknown_channel_errors(raw_root):
    _run(["init", str(raw_root)])
    assert _run(["alias", "nope", "x", "--path", str(raw_root)]) == 1


def test_alias_command_rejects_collision(raw_root):
    from apairo.core.config import read_config

    _run(["init", str(raw_root)])
    _run(["alias", "lidar", "x", "--path", str(raw_root)])
    assert _run(["alias", "imu", "x", "--path", str(raw_root)]) == 1
    # the failed command wrote nothing
    assert "alias" not in read_config(raw_root / "seq_a")["channels"]["imu"]


def test_alias_command_force_reassigns(raw_root, capsys):
    from apairo.core.config import read_config

    _run(["init", str(raw_root)])
    _run(["alias", "lidar", "x", "--path", str(raw_root)])
    capsys.readouterr()
    assert _run(["alias", "imu", "x", "--force", "--path", str(raw_root)]) == 0
    out = capsys.readouterr().out
    assert "displaced: lidar" in out
    ch = read_config(raw_root / "seq_a")["channels"]
    assert ch["imu"]["alias"] == "x" and "alias" not in ch["lidar"]


def test_status_shows_alias_root_and_sequence(raw_root, capsys):
    _run(["init", str(raw_root)])
    _run(["alias", "lidar", "points", "--path", str(raw_root)])
    capsys.readouterr()

    _run(["status", str(raw_root)])
    out = capsys.readouterr().out
    assert "aliases" in out and "lidar as points" in out   # root summary line

    capsys.readouterr()
    _run(["status", str(raw_root), "-s", "seq_a"])
    out = capsys.readouterr().out
    assert "points (lidar)" in out                          # alias-first in the table

    capsys.readouterr()
    _run(["status", str(raw_root / "seq_a"), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["channels"]["lidar"]["alias"] == "points"   # json keyed by real name
