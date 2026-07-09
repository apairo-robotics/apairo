"""The .apairo version-1 schema contract: validation of channels.yaml,
dataset.yaml (manifest), and calibration.yaml.

Policy: tolerant -- an unknown field is reported as a warning and otherwise
ignored (forward-compatible). The manifest and calibration files are optional;
their absence is not an issue.
"""

import numpy as np

from apairo.core.config import (
    SCHEMA_VERSION,
    read_manifest,
    register_static_transform,
    verify_calibration,
    verify_config,
    verify_manifest,
    write_config,
    write_manifest,
)


def _channels(root, channels):
    for key in channels:
        (root / key).mkdir(parents=True, exist_ok=True)
    write_config(root, {"version": SCHEMA_VERSION, "channels": channels})


# ── channels.yaml ────────────────────────────────────────────────────────────


def test_verify_config_valid(tmp_path):
    _channels(tmp_path, {"lidar": {"kind": "raw", "loader": "bin"}})
    assert verify_config(tmp_path) == []


def test_verify_config_unknown_field_is_a_warning(tmp_path):
    _channels(tmp_path, {"lidar": {"kind": "raw", "loader": "bin", "frmae": "ego"}})
    assert any("unknown field 'frmae'" in i for i in verify_config(tmp_path))


def test_verify_config_bad_kind(tmp_path):
    _channels(tmp_path, {"lidar": {"kind": "rawww", "loader": "bin"}})
    assert any("unknown kind 'rawww'" in i for i in verify_config(tmp_path))


def test_verify_config_transform_missing_child(tmp_path):
    _channels(
        tmp_path,
        {"odom": {"kind": "raw", "loader": "npy", "transform": {"parent": "map"}}},
    )
    assert any("transform is missing 'child'" in i for i in verify_config(tmp_path))


def test_verify_config_suffixed_channel_resolves_directory(tmp_path):
    # lidar_intensity has no directory of its own -- it shares lidar's.
    (tmp_path / "lidar").mkdir()
    write_config(
        tmp_path,
        {
            "version": SCHEMA_VERSION,
            "channels": {
                "lidar": {"kind": "raw", "loader": "npys"},
                "lidar_intensity": {
                    "kind": "raw",
                    "loader": "npys",
                    "directory": "lidar",
                    "suffix": "intensity",
                },
            },
        },
    )
    assert verify_config(tmp_path) == []


def test_verify_config_suffixed_channel_missing_directory(tmp_path):
    write_config(
        tmp_path,
        {
            "version": SCHEMA_VERSION,
            "channels": {
                "lidar_intensity": {
                    "kind": "raw",
                    "loader": "npys",
                    "directory": "lidar",
                    "suffix": "intensity",
                },
            },
        },
    )
    assert any("directory not found" in i for i in verify_config(tmp_path))


# ── dataset.yaml (manifest) -- optional ──────────────────────────────────────


def test_verify_manifest_absent_is_ok(tmp_path):
    assert verify_manifest(tmp_path) == []


def test_write_manifest_stamps_version(tmp_path):
    write_manifest(tmp_path, {"class": "Foo"})
    assert read_manifest(tmp_path)["version"] == SCHEMA_VERSION
    assert verify_manifest(tmp_path) == []


def test_verify_manifest_unknown_field_is_a_warning(tmp_path):
    write_manifest(tmp_path, {"class": "Foo", "bogus": 1})
    assert any("unknown field 'bogus'" in i for i in verify_manifest(tmp_path))


# ── calibration.yaml -- optional ─────────────────────────────────────────────


def test_verify_calibration_absent_is_ok(tmp_path):
    assert verify_calibration(tmp_path) == []


def test_verify_calibration_valid(tmp_path):
    register_static_transform(tmp_path, "lidar", "camera", np.eye(4))
    assert verify_calibration(tmp_path) == []


def test_verify_calibration_bad_matrix(tmp_path):
    (tmp_path / ".apairo").mkdir()
    (tmp_path / ".apairo" / "calibration.yaml").write_text(
        "version: 1\n"
        "transforms:\n"
        "  a_to_b:\n"
        "    parent: a\n"
        "    child: b\n"
        "    matrix: [1, 2, 3]\n"
    )
    assert any("not 4x4" in i for i in verify_calibration(tmp_path))


def test_verify_calibration_valid_camera(tmp_path):
    from apairo.core.config import register_intrinsics

    register_intrinsics(
        tmp_path,
        "cam",
        K=np.eye(3),
        distortion=[0.1, 0.0, 0.0, 0.0, 0.0],
        width=640,
        height=480,
    )
    assert verify_calibration(tmp_path) == []


def test_verify_calibration_bad_camera(tmp_path):
    (tmp_path / ".apairo").mkdir()
    (tmp_path / ".apairo" / "calibration.yaml").write_text(
        "version: 1\n"
        "cameras:\n"
        "  cam_bad_k:\n"
        "    K: [1, 2, 3]\n"
        "  cam_no_k:\n"
        "    D: [0.1]\n"
        "  cam_unknown_field:\n"
        "    K: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]\n"
        "    focal: 5\n"
    )
    issues = verify_calibration(tmp_path)
    assert any("cam_bad_k" in i and "not 3x3" in i for i in issues)
    assert any("cam_no_k" in i and "missing 'K'" in i for i in issues)
    assert any("cam_unknown_field" in i and "focal" in i for i in issues)
