r"""Prototype: a channel whose alignment key is parsed from its own filenames.

The key (a timestamp or an index) is computed in memory at read time -- **no
``timestamps.txt`` is ever written** -- and feeds ``synchronize()`` unchanged. The
key regex also drives enumeration, so names the default frame-file convention
rejects (a ``_`` in a Rellis ``frame<N>-<epoch>_<ms>`` stem) still load.
"""

from __future__ import annotations

import numpy as np
import pytest

from apairo.core.config import write_config
from apairo.dataset.raw import RawDataset

TS_KEY = r"frame\d+-(\d+)_(\d+)"  # frame<N>-<sec>_<ms> -> <sec>.<ms>
IDX_KEY = r"frame(\d+)-"  # frame<N>- -> <N>


def _frames(dirpath, names):
    dirpath.mkdir(parents=True)
    for name in names:
        np.save(dirpath / name, np.zeros((2, 3), dtype=np.float32))


def _build(root):
    """A sequence: lidar (own timestamps.txt) + a Rellis-style camera whose key is
    encoded in its filenames (sec=1000, ms=i*100), with a '_' in every name."""
    _frames(root / "lidar", [f"{i:06d}.npy" for i in range(10)])
    np.savetxt(root / "lidar" / "timestamps.txt", 1000.0 + np.arange(10) / 10.0)
    _frames(
        root / "camera", [f"frame{i:06d}-1000_{i * 100:03d}.npy" for i in range(10)]
    )
    write_config(
        root,
        {
            "version": 1,
            "channels": {
                "lidar": {"kind": "raw", "loader": "npys"},
                "camera": {"kind": "raw", "loader": "npys", "key": {"name": TS_KEY}},
            },
        },
    )
    return root


def test_key_parsed_in_memory_and_nothing_written(tmp_path):
    root = _build(tmp_path / "seq")
    ds = RawDataset(root, keys=["camera"])
    np.testing.assert_allclose(ds.timestamps["camera"], 1000.0 + np.arange(10) / 10.0)
    assert not (root / "camera" / "timestamps.txt").exists()  # read-only, in memory


def test_underscore_names_still_enumerate(tmp_path):
    # every camera stem carries a '_' (1000_000); the default frame-file convention
    # would skip them -- the key regex enumerates them instead.
    root = _build(tmp_path / "seq")
    assert len(RawDataset(root, keys=["camera"])) == 10


def test_synchronize_on_a_filename_key(tmp_path):
    root = _build(tmp_path / "seq")
    ds = RawDataset(root, keys=["lidar", "camera"])
    view = ds.synchronize(reference="lidar", method="nearest", tolerance=0.05)
    assert view.is_synchronous
    assert len(view) == 10
    assert set(view[0].data) == {"lidar", "camera"}


def test_index_key_single_group(tmp_path):
    root = tmp_path / "seq"
    _frames(root / "cam", [f"frame{i:06d}-whatever.npy" for i in range(5)])
    write_config(
        root,
        {
            "version": 1,
            "channels": {
                "cam": {"kind": "raw", "loader": "npys", "key": {"name": IDX_KEY}}
            },
        },
    )
    ds = RawDataset(root, keys=["cam"])
    np.testing.assert_array_equal(ds.timestamps["cam"], np.arange(5, dtype=float))


def test_sparse_subset_synchronizes_to_labeled_frames(tmp_path):
    # camera dense 0..5, labels only on even frames -> the sync keeps exactly those.
    root = tmp_path / "seq"
    _frames(root / "camera", [f"frame{i:06d}-x.npy" for i in range(6)])
    _frames(root / "labels", [f"frame{i:06d}-x.npy" for i in (0, 2, 4)])
    write_config(
        root,
        {
            "version": 1,
            "channels": {
                "camera": {"kind": "raw", "loader": "npys", "key": {"name": IDX_KEY}},
                "labels": {"kind": "raw", "loader": "npys", "key": {"name": IDX_KEY}},
            },
        },
    )
    ds = RawDataset(root, keys=["camera", "labels"])
    view = ds.synchronize(reference="camera", method="nearest", tolerance=0.0)
    assert len(view) == 3
    assert all(set(view[i].data) == {"camera", "labels"} for i in range(len(view)))


def test_key_regex_matching_nothing_is_a_clear_error(tmp_path):
    root = tmp_path / "seq"
    _frames(root / "cam", ["000000.npy"])  # no 'frame' prefix -> regex matches nothing
    write_config(
        root,
        {
            "version": 1,
            "channels": {
                "cam": {"kind": "raw", "loader": "npys", "key": {"name": IDX_KEY}}
            },
        },
    )
    with pytest.raises(FileNotFoundError, match="match key regex"):
        RawDataset(root, keys=["cam"])
