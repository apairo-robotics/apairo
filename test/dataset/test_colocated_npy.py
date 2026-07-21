"""Colocated stacked ``.npy`` channels -- two whole arrays in one directory.

The whole-array analogue of the per-frame ``_intensity`` idiom: a ``gicp_poses/``
directory holds ``poses.npy`` (the base) *and* ``valid_mask.npy`` (a per-pose
mask), each exposed as its own ``npy`` channel via an ``array_file`` selector.
The mask's filename is externally frozen (a KISS-ICP producer), so it cannot
follow ``<base>_<suffix>.npy`` -- ``array_file`` names it verbatim, and the name
lives in ``channels.yaml``, never in the loader.
"""

import numpy as np

from apairo.core.config import verify_config, write_config
from apairo.dataset.raw import RawDataset


def _make_gicp_poses(seq_dir, *, base_array_file=True):
    """A gicp_poses/ directory colocating poses.npy + valid_mask.npy on one clock,
    plus a channels.yaml declaring both as npy channels over the shared directory."""
    n = 4
    poses = np.arange(n * 16, dtype=float).reshape(n, 4, 4)
    mask = np.array([True, False, True, True])
    ts = np.linspace(0.0, 0.3, n)

    d = seq_dir / "gicp_poses"
    d.mkdir(parents=True)
    np.save(d / "poses.npy", poses)
    np.save(d / "valid_mask.npy", mask)
    np.savetxt(d / "timestamps.txt", ts)

    poses_entry = {"kind": "raw", "loader": "npy"}
    if base_array_file:  # declared name vs. the sort-accident default
        poses_entry["array_file"] = "poses.npy"
    write_config(
        seq_dir,
        {
            "version": 1,
            "channels": {
                "gicp_poses": poses_entry,
                "gicp_valid_mask": {
                    "kind": "raw",
                    "loader": "npy",
                    "directory": "gicp_poses",
                    "array_file": "valid_mask.npy",
                },
            },
        },
    )
    return poses, mask, ts


def test_two_colocated_arrays_load_as_separate_channels(tmp_path):
    seq = tmp_path / "seq_a"
    poses, mask, _ = _make_gicp_poses(seq)

    ds = RawDataset(seq, keys=["gicp_poses", "gicp_valid_mask"])
    assert set(ds.keys) == {"gicp_poses", "gicp_valid_mask"}

    # Each channel reads its own stacked file, row per frame.
    assert len(ds.loaders["gicp_poses"]) == len(poses)
    assert len(ds.loaders["gicp_valid_mask"]) == len(mask)
    np.testing.assert_array_equal(ds.loaders["gicp_poses"][2], poses[2])
    np.testing.assert_array_equal(ds.loaders["gicp_valid_mask"][2], mask[2])
    # The mask keeps its own dtype -- it is a real second array, not poses.
    assert ds.loaders["gicp_valid_mask"][1].dtype == np.bool_


def test_colocated_channels_share_the_directory_clock(tmp_path):
    seq = tmp_path / "seq_a"
    _make_gicp_poses(seq)
    ds = RawDataset(seq, keys=["gicp_poses", "gicp_valid_mask"])
    np.testing.assert_array_equal(
        ds.timestamps["gicp_poses"], ds.timestamps["gicp_valid_mask"]
    )


def test_synchronize_exposes_both_arrays(tmp_path):
    seq = tmp_path / "seq_a"
    poses, mask, _ = _make_gicp_poses(seq)
    ds = RawDataset(seq, keys=["gicp_poses", "gicp_valid_mask"])

    sync = ds.synchronize(reference="gicp_poses")
    np.testing.assert_array_equal(sync[2].data["gicp_poses"], poses[2])
    np.testing.assert_array_equal(sync[2].data["gicp_valid_mask"], mask[2])


def test_bare_base_still_loads_via_sort_when_only_sibling_declares_file(tmp_path):
    # The base channel omits array_file: glob picks poses.npy ('.' < '_'), while
    # the sibling names valid_mask.npy explicitly. Both remain reachable.
    seq = tmp_path / "seq_a"
    poses, mask, _ = _make_gicp_poses(seq, base_array_file=False)
    ds = RawDataset(seq, keys=["gicp_poses", "gicp_valid_mask"])
    np.testing.assert_array_equal(ds.loaders["gicp_poses"][0], poses[0])
    np.testing.assert_array_equal(ds.loaders["gicp_valid_mask"][0], mask[0])


def test_stacked_channels_report_zero_padded_stems(tmp_path):
    # array_file selects a stacked array -- no per-frame filenames, so provenance
    # stems stay the zero-padded row index (the loader gets no `.files`).
    seq = tmp_path / "seq_a"
    _make_gicp_poses(seq)
    ds = RawDataset(seq, keys=["gicp_poses", "gicp_valid_mask"])
    assert all(s.isdigit() and len(s) == 6 for s in ds.frame_stems)


# ────────────────────────────── config checks ────────────────────────────────


def test_verify_config_accepts_colocated_arrays(tmp_path):
    seq = tmp_path / "seq_a"
    _make_gicp_poses(seq)
    assert verify_config(seq) == []


def test_verify_config_flags_missing_array_file(tmp_path):
    seq = tmp_path / "seq_a"
    _make_gicp_poses(seq)
    write_config(
        seq,
        {
            "version": 1,
            "channels": {
                "gicp_poses": {"kind": "raw", "loader": "npy"},
                "gicp_valid_mask": {
                    "kind": "raw",
                    "loader": "npy",
                    "directory": "gicp_poses",
                    "array_file": "absent.npy",
                },
            },
        },
    )
    issues = verify_config(seq)
    assert any("array_file 'absent.npy' not found" in i for i in issues)


def test_verify_config_flags_array_file_on_non_npy_loader(tmp_path):
    seq = tmp_path / "seq_a"
    _make_gicp_poses(seq)
    write_config(
        seq,
        {
            "version": 1,
            "channels": {
                "gicp_poses": {
                    "kind": "raw",
                    "loader": "bin",
                    "array_file": "poses.npy",
                },
            },
        },
    )
    issues = verify_config(seq)
    assert any("only meaningful for the 'npy' loader" in i for i in issues)


# ─────────────────────────────── status (cli) ────────────────────────────────


def test_status_resolves_array_file_shape(tmp_path):
    from apairo.cli import _channel_detail

    seq = tmp_path / "seq_a"
    _make_gicp_poses(seq)
    meta = {
        "kind": "raw",
        "loader": "npy",
        "directory": "gicp_poses",
        "array_file": "valid_mask.npy",
    }
    detail = _channel_detail(seq, "gicp_valid_mask", meta)
    # The mask, not poses (glob[0]): a per-frame scalar (shape []) of dtype bool.
    assert detail["shape"] == []
    assert detail["dtype"] == "bool"
    assert detail["frames"] == 4
