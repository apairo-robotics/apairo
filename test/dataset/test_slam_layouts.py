"""The TUM RGB-D and EuRoC MAV layouts, read from a declaration only.

The fixtures under ``test/assets/mini_tum`` and ``test/assets/mini_euroc`` are
synthetic (``make_slam_fixtures.py``) but reproduce each dataset's tree, file
naming, header lines and timestamp units. The declarations are the ones that
ship with apairo (``apairo/dataset/declarations/``, selected by name:
``declare="tum_rgbd"``) and are shown in the dataset guides, so a drift between
a guide and the loader fails here.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from apairo.cli import main
from apairo.dataset.raw import RawDataset
from apairo.dataset.registry import DECLARATIONS_DIR

ASSETS = Path(__file__).parent.parent / "assets"
DECLARATIONS = DECLARATIONS_DIR

EUROC = [
    ("V1_01_easy", "euroc_vicon_room.yaml", "vicon0", 7),
    ("MH_05_difficult", "euroc_machine_hall.yaml", "leica0", 3),
]


@pytest.fixture
def tum(tmp_path):
    seq = tmp_path / "rgbd_dataset_freiburg1_xyz"
    shutil.copytree(ASSETS / "mini_tum" / "rgbd_dataset_freiburg1_xyz", seq)
    return seq


def _euroc(tmp_path, name):
    seq = tmp_path / name / "mav0"
    shutil.copytree(ASSETS / "mini_euroc" / name / "mav0", seq)
    return seq


def _cli(args, capsys):
    try:
        code = main(args)
    except SystemExit as exc:
        code = exc.code
    return code, capsys.readouterr().out


# ───────────────────────────────── TUM RGB-D ─────────────────────────────────


def test_tum_loads_from_the_declaration(tum):
    ds = RawDataset(tum, declare=DECLARATIONS / "tum_rgbd.yaml")
    assert ds.keys == ["accelerometer", "depth", "groundtruth", "rgb"]
    assert ds.shape == {
        "accelerometer": (3,),
        "depth": (6, 8),
        "groundtruth": (7,),
        "rgb": (6, 8, 3),
    }
    # Image clocks come from <sec>.<usec> in the filenames, the tables' from
    # their first column: both land on the same epoch, in seconds.
    assert ds.timestamps["rgb"][0] == pytest.approx(1305031102.175304, abs=1e-6)
    assert ds.timestamps["groundtruth"][0] == pytest.approx(1305031102.1553, abs=1e-4)


def test_tum_association_drops_the_colour_frame_with_no_depth_in_tolerance(tum):
    ds = RawDataset(tum, declare=DECLARATIONS / "tum_rgbd.yaml")
    frames = ds.synchronize(reference="rgb", method="nearest", tolerance=0.02)
    # The third depth image is 30 ms off its colour frame (see the generator).
    assert len(frames) == 5
    assert 2 not in frames.frame_indices["rgb"]
    assert np.abs(frames.time_offsets("depth")).max() <= 0.02
    assert frames[0].data["depth"].dtype == np.uint16


def test_tum_data_files_are_left_untouched(tum):
    before = {p: p.read_bytes() for p in tum.rglob("*") if p.is_file()}
    RawDataset(tum, declare=DECLARATIONS / "tum_rgbd.yaml")[0]
    for path, content in before.items():
        assert path.read_bytes() == content
    added = {p.relative_to(tum).parts[0] for p in tum.rglob("*")} - {
        p.relative_to(tum).parts[0] for p in before
    }
    assert added <= {".apairo"}  # apairo's own registry, nothing else


# ──────────────────────────────── EuRoC MAV ──────────────────────────────────


@pytest.mark.parametrize("name, declaration, tracker, width", EUROC)
def test_euroc_loads_from_the_hall_declaration(
    tmp_path, name, declaration, tracker, width
):
    ds = RawDataset(_euroc(tmp_path, name), declare=DECLARATIONS / declaration)
    assert ds.keys == sorted(["cam0", "cam1", "groundtruth", "imu0", tracker])
    assert ds.shape["cam0"] == (6, 8)
    assert ds.shape["imu0"] == (6,)
    assert ds.shape["groundtruth"] == (16,)
    assert ds.shape[tracker] == (width,)
    # Nanosecond stamps, in the filenames and in the tables, fold to seconds.
    assert np.diff(ds.timestamps["cam0"]) == pytest.approx(0.05, abs=1e-6)
    assert np.diff(ds.timestamps["imu0"]) == pytest.approx(0.005, abs=1e-6)


@pytest.mark.parametrize("name, declaration, tracker, width", EUROC)
def test_euroc_imu_and_state_align_onto_the_camera(
    tmp_path, name, declaration, tracker, width
):
    ds = RawDataset(
        _euroc(tmp_path, name),
        keys=["cam0", "cam1", "imu0", "groundtruth"],
        declare=DECLARATIONS / declaration,
    )
    frames = ds.synchronize(reference="cam0", method="nearest", tolerance=0.005)
    # The state starts two camera frames late: those frames are dropped.
    assert len(frames) == 3
    np.testing.assert_array_equal(frames.frame_indices["cam0"], [2, 3, 4])
    for i, cam_row in enumerate([2, 3, 4]):  # camera frame r sits on IMU row 10 * r
        assert frames[i].data["imu0"][0] == 10.0 * cam_row
    assert np.abs(frames.time_offsets("groundtruth")).max() < 1e-6


# ─────────────────────── the CLI reads through --declare ──────────────────────


def test_status_and_check_read_the_layout_through_an_external_declaration(tum, capsys):
    """An external declaration is what loading reads the tree through
    (``declare=``); ``status`` and ``check`` must read it the same way, or they
    report channels as clockless that load fine."""
    declaration = str(DECLARATIONS / "tum_rgbd.yaml")
    RawDataset(tum, declare=declaration)  # bootstraps the registry (rgb, depth)

    code, out = _cli(["check", str(tum), "--declare", declaration, "--json"], capsys)
    assert code == 0, out
    assert json.loads(out) == {"ok": True, "issues": []}

    _, out = _cli(["status", str(tum), "--declare", declaration, "--json"], capsys)
    channels = json.loads(out)["channels"]
    assert sorted(channels) == ["accelerometer", "depth", "groundtruth", "rgb"]
    assert channels["groundtruth"]["frames"] == 30
    assert channels["groundtruth"]["rate_hz"] == pytest.approx(100.0, rel=1e-3)

    # Without the declaration the images have no clock, and check says so.
    code, out = _cli(["check", str(tum), "--json"], capsys)
    assert code == 1
    assert any("channel 'rgb': no clock" in i for i in json.loads(out)["issues"])


@pytest.mark.parametrize("name, declaration, tracker, width", EUROC)
def test_check_is_clean_on_euroc_with_its_declaration(
    tmp_path, capsys, name, declaration, tracker, width
):
    seq = _euroc(tmp_path, name)
    RawDataset(seq, declare=DECLARATIONS / declaration)
    code, out = _cli(
        ["check", str(seq), "--declare", str(DECLARATIONS / declaration)], capsys
    )
    assert code == 0, out


# ─────────────────────── shipped declarations, by name ───────────────────────


def test_a_shipped_declaration_is_selected_by_name(tum, capsys):
    ds = RawDataset(tum, declare="tum_rgbd")
    assert set(ds.keys) == {"rgb", "depth", "groundtruth", "accelerometer"}
    code, out = _cli(["check", str(tum), "--declare", "tum_rgbd"], capsys)
    assert code == 0, out


def test_an_unknown_declaration_name_lists_the_shipped_ones(tum, capsys):
    with pytest.raises(FileNotFoundError, match="tum_rgbd"):
        RawDataset(tum, declare="tum_rgbdd")
    code, out = _cli(["check", str(tum), "--declare", "tum_rgbdd"], capsys)
    assert code == 1
    assert "nor a declaration that ships with apairo: euroc_machine_hall" in out
