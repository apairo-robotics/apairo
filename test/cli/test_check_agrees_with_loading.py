"""`check` and `status` never crash on a malformed dataset, and `check` never
answers OK for one that loading refuses."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from apairo import RawDataset
from apairo.cli import main
from apairo.core.config import ConfigError


def _channel(root, name, stamps):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    for i in range(len(stamps)):
        np.save(d / f"{i:06d}.npy", np.full(2, float(i)))
    np.savetxt(d / "timestamps.txt", stamps)
    return d


def _sidecar(root, name, text):
    (root / ".apairo").mkdir(exist_ok=True)
    (root / ".apairo" / name).write_text(text)


def _unsorted(r):
    _channel(r, "lidar", [0, 0.2, 0.1])


def _two_columns(r):
    np.savetxt(
        _channel(r, "lidar", [0, 0.1, 0.2]) / "timestamps.txt",
        np.c_[[0, 0.1, 0.2], [5, 6, 7]],
    )


def _unreadable(r):
    (_channel(r, "lidar", [0, 0.1, 0.2]) / "timestamps.txt").write_text("a\nb\nc\n")


def _not_finite(r):
    _channel(r, "lidar", [0, np.nan, 0.2])


def _wrong_types(r):
    _channel(r, "lidar", [0, 0.1, 0.2])
    _sidecar(
        r,
        "channels.yaml",
        "version: 1\nchannels:\n  lidar: {loader: npys, sources: 3, "
        "timestamps_from: [a], alias: [x], kind: 7}\n",
    )


def _scalar_transform(r):
    _channel(r, "lidar", [0, 0.1, 0.2])
    _sidecar(
        r,
        "channels.yaml",
        "version: 1\nchannels:\n  lidar: {loader: npys, transform: 3}\n",
    )


def _odd_views(r):
    _channel(r, "lidar", [0, 0.1, 0.2])
    _sidecar(
        r, "views.yaml", "version: 1\nviews:\n  v: {file: 3, method: x, sources: 4}\n"
    )


def _odd_calibration(r):
    _channel(r, "lidar", [0, 0.1, 0.2])
    _sidecar(r, "calibration.yaml", "version: 1\ntransforms: [1, 2]\ncameras: 5\n")


def _bad_matrix(r):
    _channel(r, "lidar", [0, 0.1, 0.2])
    _sidecar(
        r,
        "calibration.yaml",
        "version: 1\ntransforms:\n  c_to_d: {parent: c, child: d, matrix: nope}\n",
    )


def _manifest_list(r):
    _channel(r / "seq", "lidar", [0, 0.1, 0.2])
    _sidecar(r, "dataset.yaml", "- 1\n- 2\n")


def _odd_declaration(r):
    _channel(r, "lidar", [0, 0.1, 0.2])
    (r / "apairo.yaml").write_text("version: 1\nchannels:\n  lidar: 3\n")


def _broken_yaml(r):
    _channel(r, "lidar", [0, 0.1, 0.2])
    for name in ("calibration.yaml", "views.yaml", "dataset.yaml"):
        _sidecar(r, name, "a: [\n")


CASES = {
    f.__name__.lstrip("_"): f
    for f in (
        _unsorted,
        _two_columns,
        _unreadable,
        _not_finite,
        _wrong_types,
        _scalar_transform,
        _odd_views,
        _odd_calibration,
        _bad_matrix,
        _manifest_list,
        _odd_declaration,
        _broken_yaml,
    )
}


def _run(argv):
    try:
        main(argv)
    except SystemExit as exc:
        return exc.code
    return 0


@pytest.mark.parametrize("build", CASES.values(), ids=CASES)
def test_check_and_status_agree_with_loading(tmp_path, build, capsys):
    build(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            RawDataset(tmp_path)
            refused = False
        except Exception:
            refused = True
    # Loading may have written the sidecar; check is asked about that state.
    check = _run(["check", str(tmp_path)])  # a crash would fail the test here
    capsys.readouterr()
    _run(["status", str(tmp_path)])
    _run(["status", str(tmp_path), "--json"])
    capsys.readouterr()
    assert check in (0, 1)
    if refused:
        assert check == 1  # never OK for a dataset that does not load


def test_check_names_the_clock_problem(tmp_path, capsys):
    _unsorted(tmp_path)
    assert _run(["check", str(tmp_path), "--json"]) == 1
    issues = json.loads(capsys.readouterr().out)["issues"]
    assert any(
        "timestamps.txt is not ascending" in i and "frame 2" in i for i in issues
    )


def test_check_opens_an_initialised_dataset_as_a_last_resort(tmp_path, capsys):
    """A refusal no dedicated check covers is still reported: here, frames of a
    profiled dataset that do not line up across channels."""
    from apairo.dataset.semantic_kitti import SemanticKittiDataset

    seq = tmp_path / "sequences" / "00"
    for sub, ext, ids in (
        ("velodyne", "bin", (0, 1, 2)),
        ("labels", "label", (0, 1, 2)),
    ):
        (seq / sub).mkdir(parents=True)
        for i in ids:
            np.zeros((2, 4), np.float32).tofile(seq / sub / f"{i:06d}.{ext}")
    SemanticKittiDataset.init(tmp_path)
    assert _run(["check", str(tmp_path)]) == 0
    capsys.readouterr()
    (seq / "labels" / "000000.label").rename(seq / "labels" / "000003.label")
    assert _run(["check", str(tmp_path)]) == 1
    assert "loading refuses this dataset" in capsys.readouterr().out


def test_a_manifest_cannot_list_a_sequence_outside_the_root(tmp_path):
    outside = tmp_path / "outside"
    _channel(outside, "lidar", [0, 0.1])
    root = tmp_path / "root"
    _channel(root / "seq", "lidar", [0, 0.1])
    _sidecar(root, "dataset.yaml", "version: 1\nsequences: [seq, ../outside]\n")
    with pytest.raises(ValueError, match="relative path"):
        RawDataset(root)
    _sidecar(root, "dataset.yaml", "version: 1\nsequences: seq\n")
    with pytest.raises(ConfigError, match="list of directory names"):
        RawDataset(root)
