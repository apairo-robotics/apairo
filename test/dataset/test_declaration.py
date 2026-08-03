"""Tests for the human-owned declaration (``apairo.yaml`` / ``declare=``).

The declaration is the *human* half of the channel metadata -- how to read a
directory (loaders, aliases, ``key`` specs, ``fields`` contracts) -- overlaid
per channel and per field onto the machine registry (``.apairo/channels.yaml``).
Ownership is the contract under test: apairo never writes a declaration, so
``init --overwrite`` cannot destroy it, and machine provenance (``kind:
preprocess``, ``sources``) is refused inside one.
"""

from __future__ import annotations

import os
import stat

import numpy as np
import pytest
import yaml

from apairo.core.config import (
    CONFIG_DIR,
    declaration_path,
    read_config,
    read_declaration,
    verify_declaration,
    write_config,
)
from apairo.dataset.raw import RawDataset


def _frames(dirpath, names):
    dirpath.mkdir(parents=True)
    for name in names:
        np.save(dirpath / name, np.zeros((2, 3), dtype=np.float32))


def _declare(path, channels):
    with open(path, "w") as f:
        yaml.dump({"version": 1, "channels": channels}, f)
    return path


def _make_seq(root, n=5):
    """A bare sequence (no .apairo): one npys ``lidar`` with a timestamps.txt."""
    _frames(root / "lidar", [f"{i:06d}.npy" for i in range(n)])
    np.savetxt(root / "lidar" / "timestamps.txt", np.arange(n, dtype=float))
    return root


# ─────────────────────────────── overlay ─────────────────────────────────────


def test_declaration_overlays_discovered_channel(tmp_path):
    root = _make_seq(tmp_path / "seq")
    _declare(root / "apairo.yaml", {"lidar": {"alias": "cloud"}})
    ds = RawDataset(root)
    assert ds.keys == ["cloud"]
    assert ds[0].data["cloud"].shape == (2, 3)


def test_registry_never_carries_declaration_content(tmp_path):
    # The bootstrap writes the machine registry; the declared alias stays out
    # of it -- .apairo records what exists, apairo.yaml how to read it.
    root = _make_seq(tmp_path / "seq")
    _declare(root / "apairo.yaml", {"lidar": {"alias": "cloud"}})
    RawDataset(root)
    registry = read_config(root)["channels"]
    assert "alias" not in registry["lidar"]


def test_declaration_key_spec_drives_enumeration_and_clock(tmp_path):
    # barakuda-style stems: <scene>_<ns>. The '_' would be skipped by the
    # default frame-file convention; the declared key regex enumerates them
    # and parses the clock, in memory.
    root = tmp_path / "seq"
    stems = [f"scene_{int(t * 1e9)}" for t in (1.0, 2.0, 3.0)]
    _frames(root / "lidar", [f"{s}.npy" for s in stems])
    _declare(
        root / "apairo.yaml",
        {"lidar": {"key": {"name": r"(\d+)$", "units": ["ns"]}}},
    )
    ds = RawDataset(root)
    assert len(ds) == 3
    np.testing.assert_allclose(ds.timestamps["lidar"], [1.0, 2.0, 3.0])
    assert not (root / "lidar" / "timestamps.txt").exists()


def test_declaration_adds_channel_missing_from_registry(tmp_path):
    root = _make_seq(tmp_path / "seq")
    write_config(
        root, {"version": 1, "channels": {"lidar": {"kind": "raw", "loader": "npys"}}}
    )
    _frames(root / "imu", [f"{i:06d}.npy" for i in range(5)])
    np.savetxt(root / "imu" / "timestamps.txt", np.arange(5, dtype=float))
    _declare(root / "apairo.yaml", {"imu": {"loader": "npys"}})
    ds = RawDataset(root)
    assert set(ds.keys) == {"lidar", "imu"}


def test_declare_param_overrides_in_tree_per_field(tmp_path):
    # In-tree declares a key spec + an alias; declare= overrides the alias
    # only. Per-field merge: the key spec must survive.
    root = tmp_path / "seq"
    _frames(root / "lidar", [f"scene_{i}000000000.npy" for i in (1, 2, 3)])
    _declare(
        root / "apairo.yaml",
        {"lidar": {"alias": "cloud", "key": {"name": r"(\d+)$", "units": ["ns"]}}},
    )
    external = _declare(tmp_path / "eval.yaml", {"lidar": {"alias": "pts"}})
    ds = RawDataset(root, declare=external)
    assert ds.keys == ["pts"]
    np.testing.assert_allclose(ds.timestamps["pts"], [1.0, 2.0, 3.0])


# ─────────────────────────────── ownership ───────────────────────────────────


def test_declared_only_channel_requires_loader(tmp_path):
    root = _make_seq(tmp_path / "seq")
    _declare(root / "apairo.yaml", {"ghost": {"alias": "boo"}})
    with pytest.raises(ValueError, match="loader"):
        RawDataset(root)


def test_declaration_refuses_preprocess_kind(tmp_path):
    path = _declare(
        tmp_path / "apairo.yaml", {"trav": {"kind": "preprocess", "loader": "npys"}}
    )
    with pytest.raises(ValueError, match="preprocess"):
        read_declaration(path)


def test_declaration_refuses_machine_provenance(tmp_path):
    path = _declare(
        tmp_path / "apairo.yaml",
        {"trav": {"loader": "npys", "sources": ["lidar"]}},
    )
    with pytest.raises(ValueError, match="sources"):
        read_declaration(path)


def test_init_overwrite_preserves_declaration(tmp_path):
    # The point of the ownership split: a full registry rebuild cannot touch
    # the human file, and the declaration still applies after it.
    root = _make_seq(tmp_path / "seq")
    decl = _declare(root / "apairo.yaml", {"lidar": {"alias": "cloud"}})
    RawDataset(root)
    before = decl.read_bytes()
    RawDataset.init(root, overwrite=True)
    assert decl.read_bytes() == before
    assert RawDataset(root).keys == ["cloud"]


def test_init_respects_declared_key_no_suffix_fanout(tmp_path):
    # Underscore stems explained by a declared key regex: neither the fresh
    # scan nor a later merge may fan them out into suffixed sub-channels.
    root = tmp_path / "seq"
    _frames(root / "lidar", [f"scene_{i}000000000.npy" for i in (1, 2, 3)])
    _declare(
        root / "apairo.yaml",
        {"lidar": {"key": {"name": r"(\d+)$", "units": ["ns"]}}},
    )
    RawDataset.init(root)
    assert set(read_config(root)["channels"]) == {"lidar"}
    with pytest.raises(ValueError, match="No new recognizable channels"):
        RawDataset.init(root, merge=True)
    assert set(read_config(root)["channels"]) == {"lidar"}


def test_no_declaration_is_ever_written(tmp_path):
    root = _make_seq(tmp_path / "seq")
    RawDataset(root)
    RawDataset.init(root, overwrite=True)
    assert not (root / "apairo.yaml").exists()


def test_readonly_tree_loads_with_external_declare(tmp_path):
    # Zero writes: no .apairo can be written (read-only mount), the
    # declaration lives outside the tree, loading still works in full.
    root = _make_seq(tmp_path / "seq")
    external = _declare(tmp_path / "eval.yaml", {"lidar": {"alias": "cloud"}})
    ro = stat.S_IRUSR | stat.S_IXUSR
    os.chmod(root, ro)
    os.chmod(root / "lidar", ro)
    try:
        ds = RawDataset(root, declare=external)
        assert ds.keys == ["cloud"]
        assert not (root / CONFIG_DIR).exists()
    finally:
        rw = ro | stat.S_IWUSR
        os.chmod(root / "lidar", rw)
        os.chmod(root, rw)


# ─────────────────────────────── root ────────────────────────────────────────


def test_root_propagates_declare_to_sequences(tmp_path):
    root = tmp_path / "root"
    for seq in ("seq_a", "seq_b"):
        _make_seq(root / seq, n=3)
    external = _declare(tmp_path / "eval.yaml", {"lidar": {"alias": "cloud"}})
    ds = RawDataset(root, declare=external)
    assert len(ds) == 6
    assert ds.keys == ["cloud"]


# ─────────────────────────────── verify ──────────────────────────────────────


def test_verify_declaration_flags_typos_and_ownership(tmp_path):
    root = _make_seq(tmp_path / "seq")
    _declare(
        root / "apairo.yaml",
        {
            "lidar": {"feilds": ["x", "y"]},
            "trav": {"kind": "preprocess", "loader": "npys"},
        },
    )
    issues = verify_declaration(declaration_path(root), root)
    assert any("feilds" in i for i in issues)
    assert any("preprocess" in i for i in issues)


def test_verify_declaration_absent_is_fine(tmp_path):
    assert verify_declaration(tmp_path / "apairo.yaml") == []


def test_unknown_field_tolerated_at_load(tmp_path):
    root = _make_seq(tmp_path / "seq")
    _declare(root / "apairo.yaml", {"lidar": {"alias": "cloud", "notafield": 1}})
    assert RawDataset(root).keys == ["cloud"]
