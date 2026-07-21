r"""Prototype: a channel whose alignment key is parsed from its own filenames.

The key (a timestamp or an index) is computed in memory at read time -- **no
``timestamps.txt`` is ever written** -- and feeds ``synchronize()`` unchanged. The
key regex also drives enumeration, so names the default frame-file convention
rejects (a ``_`` in a Rellis ``frame<N>-<epoch>_<ms>`` stem) still load.
"""

from __future__ import annotations

import numpy as np
import pytest

from apairo.core.config import verify_config, write_config
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
    _write(root, {"cam": {"kind": "raw", "loader": "npys", "key": {"name": IDX_KEY}}})
    with pytest.raises(FileNotFoundError, match="enumeration regex"):
        RawDataset(root, keys=["cam"])


# ── scale, sidecar file, order-only, callable, guards ─────────────────────────


def _write(root, channels):
    write_config(root, {"version": 1, "channels": channels})


def test_scale_combines_groups(tmp_path):
    # unpadded ms -> the join-dot rule would mis-scale; `scale` is explicit.
    root = tmp_path / "seq"
    _frames(root / "cam", [f"frame{i:06d}-1000_{i * 10}.npy" for i in range(4)])
    _write(
        root,
        {
            "cam": {
                "kind": "raw",
                "loader": "npys",
                "key": {"name": TS_KEY, "scale": [1, 0.001]},
            }
        },
    )
    ds = RawDataset(root, keys=["cam"])
    np.testing.assert_allclose(
        ds.timestamps["cam"], [1000.0, 1000.01, 1000.02, 1000.03]
    )


def test_scale_length_mismatch_errors(tmp_path):
    root = tmp_path / "seq"
    _frames(root / "cam", [f"frame{i:06d}-1000_0.npy" for i in range(2)])
    _write(
        root,
        {
            "cam": {
                "kind": "raw",
                "loader": "npys",
                "key": {"name": TS_KEY, "scale": [1]},
            }
        },
    )  # 1 scale, 2 groups
    with pytest.raises(ValueError, match="scale"):
        RawDataset(root, keys=["cam"])


def test_sidecar_key_file(tmp_path):
    # a differently-named timestamps file; default (numeric) enumeration is fine.
    root = tmp_path / "seq"
    _frames(root / "cam", [f"{i:06d}.npy" for i in range(4)])
    np.savetxt(root / "cam" / "clock.txt", [10.0, 11.0, 12.0, 13.0])
    _write(
        root, {"cam": {"kind": "raw", "loader": "npys", "key": {"file": "clock.txt"}}}
    )
    np.testing.assert_array_equal(
        RawDataset(root, keys=["cam"]).timestamps["cam"], [10.0, 11.0, 12.0, 13.0]
    )


def test_order_only_enumerates_key_from_timestamps(tmp_path):
    # `order` (no key): enumerate the '_'-named files; key still from timestamps.txt.
    root = tmp_path / "seq"
    _frames(root / "cam", [f"frame{i:06d}-x_y.npy" for i in range(4)])
    np.savetxt(root / "cam" / "timestamps.txt", [5.0, 6.0, 7.0, 8.0])
    _write(root, {"cam": {"kind": "raw", "loader": "npys", "order": {"name": IDX_KEY}}})
    ds = RawDataset(root, keys=["cam"])
    assert len(ds) == 4
    np.testing.assert_array_equal(ds.timestamps["cam"], [5.0, 6.0, 7.0, 8.0])


def test_key_spec_beats_on_disk_timestamps(tmp_path):
    root = tmp_path / "seq"
    _frames(root / "cam", [f"frame{i:06d}-x.npy" for i in range(3)])
    np.savetxt(root / "cam" / "timestamps.txt", [99.0, 98.0, 97.0])  # ignored
    _write(root, {"cam": {"kind": "raw", "loader": "npys", "key": {"name": IDX_KEY}}})
    np.testing.assert_array_equal(
        RawDataset(root, keys=["cam"]).timestamps["cam"], [0.0, 1.0, 2.0]
    )


def test_out_of_order_files_sorted_by_key(tmp_path):
    root = tmp_path / "seq"
    # captured key descends with frame number -> enumerated in KEY order, not name order.
    _frames(
        root / "cam", ["frame000000-30.npy", "frame000001-20.npy", "frame000002-10.npy"]
    )
    _write(
        root,
        {"cam": {"kind": "raw", "loader": "npys", "key": {"name": r"frame\d+-(\d+)"}}},
    )
    np.testing.assert_array_equal(
        RawDataset(root, keys=["cam"]).timestamps["cam"], [10.0, 20.0, 30.0]
    )


def test_non_monotonic_key_provider_rejected(tmp_path):
    # the callable path has no sort -- a non-monotonic array is caught at construction.
    root = tmp_path / "seq"
    _frames(root / "cam", [f"{i:06d}.npy" for i in range(3)])
    _write(root, {"cam": {"kind": "raw", "loader": "npys"}})
    with pytest.raises(ValueError, match="non-decreasing"):
        _CallableKeyDataset(
            root,
            keys=["cam"],
            key_providers={"cam": lambda f: np.array([3.0, 1.0, 2.0])},
        )


class _CallableKeyDataset(RawDataset):
    def __init__(self, directory, keys=None, key_providers=None):
        self._key_providers = key_providers or {}
        super().__init__(directory, keys=keys)


def test_callable_key_provider(tmp_path):
    root = tmp_path / "seq"
    _frames(root / "cam", [f"{i:06d}.npy" for i in range(4)])
    _write(root, {"cam": {"kind": "raw", "loader": "npys"}})  # no key spec
    ds = _CallableKeyDataset(
        root,
        keys=["cam"],
        key_providers={"cam": lambda files: np.arange(len(files)) * 2.0},
    )
    np.testing.assert_array_equal(ds.timestamps["cam"], [0.0, 2.0, 4.0, 6.0])


def test_callable_key_provider_overrides_yaml(tmp_path):
    root = tmp_path / "seq"
    _frames(root / "cam", [f"frame{i:06d}-x.npy" for i in range(3)])
    _write(root, {"cam": {"kind": "raw", "loader": "npys", "key": {"name": IDX_KEY}}})
    ds = _CallableKeyDataset(
        root,
        keys=["cam"],
        key_providers={"cam": lambda files: np.arange(len(files)) + 100.0},
    )
    np.testing.assert_array_equal(
        ds.timestamps["cam"], [100.0, 101.0, 102.0]
    )  # callable wins


# ── hardening (adversarial edge cases) ────────────────────────────────────────


def test_non_zero_padded_index_ordered_numerically(tmp_path):
    root = tmp_path / "seq"
    # lexicographic would give frame1, frame10, frame2 -> non-monotonic keys.
    _frames(root / "cam", ["frame1-x.npy", "frame2-x.npy", "frame10-x.npy"])
    _write(root, {"cam": {"kind": "raw", "loader": "npys", "key": {"name": IDX_KEY}}})
    np.testing.assert_array_equal(
        RawDataset(root, keys=["cam"]).timestamps["cam"], [1.0, 2.0, 10.0]
    )


def test_stray_files_excluded_from_enumeration(tmp_path):
    root = tmp_path / "seq"
    _frames(root / "cam", [f"frame{i:06d}-{i}.npy" for i in range(3)])
    (root / "cam" / "notes2.md").write_text(
        "junk"
    )  # stem 'notes2' matches a loose regex
    np.savetxt(root / "cam" / "timestamps.txt", [9, 9, 9])  # shadowed, wrong ext anyway
    _write(
        root,
        {"cam": {"kind": "raw", "loader": "npys", "key": {"name": r"frame\d+-(\d+)"}}},
    )
    ds = RawDataset(root, keys=["cam"])
    assert len(ds) == 3  # notes2.md + timestamps.txt excluded
    np.testing.assert_array_equal(ds.timestamps["cam"], [0.0, 1.0, 2.0])


def test_more_than_two_groups_needs_scale(tmp_path):
    root = tmp_path / "seq"
    _frames(root / "cam", ["frame000000-1-2-3.npy"])
    _write(
        root,
        {
            "cam": {
                "kind": "raw",
                "loader": "npys",
                "key": {"name": r"(\d+)-(\d+)-(\d+)"},
            }
        },
    )
    with pytest.raises(ValueError, match="scale"):
        RawDataset(root, keys=["cam"])


def test_non_numeric_group_clear_error(tmp_path):
    root = tmp_path / "seq"
    _frames(root / "cam", ["camABC.npy", "camDEF.npy"])
    _write(
        root, {"cam": {"kind": "raw", "loader": "npys", "key": {"name": r"cam(.+)"}}}
    )
    with pytest.raises(ValueError, match="non-numeric"):
        RawDataset(root, keys=["cam"])


def test_verify_config_flags_bad_key_specs(tmp_path):
    root = tmp_path / "seq"
    _frames(root / "cam", ["000000.npy"])
    _write(
        root,
        {
            "cam": {
                "kind": "raw",
                "loader": "npys",
                "key": {"name": r"(\d+)-(\d+)-(\d+)"},
            }
        },
    )  # 3 groups, no scale
    assert any("scale" in i for i in verify_config(root))


def test_verify_config_flags_invalid_regex(tmp_path):
    root = tmp_path / "seq"
    _frames(root / "cam", ["000000.npy"])
    _write(
        root, {"cam": {"kind": "raw", "loader": "npys", "key": {"name": r"frame(\d+"}}}
    )
    assert any("valid regex" in i for i in verify_config(root))


# ── units: sugar over scale ───────────────────────────────────────────────────


def test_units_sugar_matches_scale(tmp_path):
    root = tmp_path / "seq"
    _frames(
        root / "cam", [f"frame{i:06d}-1000_{i * 10}.npy" for i in range(4)]
    )  # unpadded ms
    _write(
        root,
        {
            "cam": {
                "kind": "raw",
                "loader": "npys",
                "key": {"name": TS_KEY, "units": ["s", "ms"]},
            }
        },
    )
    np.testing.assert_allclose(
        RawDataset(root, keys=["cam"]).timestamps["cam"],
        [1000.0, 1000.01, 1000.02, 1000.03],
    )


def test_units_and_scale_are_mutually_exclusive(tmp_path):
    root = tmp_path / "seq"
    _frames(root / "cam", [f"frame{i:06d}-1000_0.npy" for i in range(2)])
    _write(
        root,
        {
            "cam": {
                "kind": "raw",
                "loader": "npys",
                "key": {"name": TS_KEY, "units": ["s", "ms"], "scale": [1, 0.001]},
            }
        },
    )
    with pytest.raises(ValueError, match="both 'units' and 'scale'"):
        RawDataset(root, keys=["cam"])


def test_unknown_unit_errors(tmp_path):
    root = tmp_path / "seq"
    _frames(root / "cam", [f"frame{i:06d}-1000_0.npy" for i in range(2)])
    _write(
        root,
        {
            "cam": {
                "kind": "raw",
                "loader": "npys",
                "key": {"name": TS_KEY, "units": ["s", "furlongs"]},
            }
        },
    )
    with pytest.raises(ValueError, match="unknown key unit"):
        RawDataset(root, keys=["cam"])


def test_verify_config_flags_units_length(tmp_path):
    root = tmp_path / "seq"
    _frames(root / "cam", ["000000.npy"])
    _write(
        root,
        {
            "cam": {
                "kind": "raw",
                "loader": "npys",
                "key": {"name": TS_KEY, "units": ["s"]},
            }
        },
    )  # 1 unit, 2 groups
    assert any("units" in i for i in verify_config(root))


def test_heterogeneous_name_captures_only_the_key(tmp_path):
    # camera_<seconds>_<frame-index>: the index is NOT the alignment key. Capture
    # only the seconds into the key; `order` uses the index for sorting. No summing
    # an index into a timestamp.
    root = tmp_path / "seq"
    _frames(
        root / "cam",
        ["camera_1000_000041.npy", "camera_1000_000042.npy", "camera_1001_000043.npy"],
    )
    _write(
        root,
        {
            "cam": {
                "kind": "raw",
                "loader": "npys",
                "key": {"name": r"camera_(\d+)_\d+", "units": ["s"]},
                "order": {"name": r"camera_\d+_(\d+)"},
            }
        },
    )
    np.testing.assert_array_equal(
        RawDataset(root, keys=["cam"]).timestamps["cam"], [1000.0, 1000.0, 1001.0]
    )
