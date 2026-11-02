"""The built-in ``hdf5`` format: a channel is one array of a container file,
its clock a sibling array -- the layout of asynchronous robot recordings."""

import pickle

import numpy as np
import pytest
import yaml

import apairo
from apairo.loader.formats import Hdf5Format
from apairo.loader.hdf5_loader import Hdf5ArrayLoader
from apairo.testing import check_format

h5py = pytest.importorskip("h5py")


def _recording(path, n_force=500, n_pose=10):
    """An asynchronous recording: two sensors, each with its own clock array."""
    with h5py.File(path, "w") as f:
        f["robot_state/force"] = np.arange(n_force * 3, dtype=np.float64).reshape(
            n_force, 3
        )
        f["robot_state/pose"] = np.arange(n_pose * 7, dtype=np.float64).reshape(
            n_pose, 7
        )
        f["timestamps/force"] = 100.0 + np.arange(n_force) * 0.002  # 500 Hz, s
        f["timestamps/pose"] = (100.0 + np.arange(n_pose) * 0.1) * 1e9  # 10 Hz, ns
        f["meta/scalar"] = 3.0
    return path


DECLARATION = {
    "version": 1,
    "channels": {
        "force": {
            "loader": "hdf5",
            "directory": ".",
            "array_file": "*.h5",
            "array": "/robot_state/force",
            "key": {"array": "/timestamps/force", "units": ["s"]},
        },
        "pose": {
            "loader": "hdf5",
            "directory": ".",
            "array_file": "*.h5",
            "array": "/robot_state/pose",
            "key": {"array": "/timestamps/pose", "units": ["ns"]},
        },
    },
}


def _sequence(tmp_path, name="demo_0", declare=True):
    seq = tmp_path / name
    seq.mkdir()
    _recording(seq / f"{name}.h5")
    if declare:
        (seq / "apairo.yaml").write_text(yaml.safe_dump(DECLARATION))
    return seq


def test_follows_the_format_contract(tmp_path):
    _recording(tmp_path / "rec.h5")
    check_format(
        Hdf5Format(),
        tmp_path,
        {"array": "/robot_state/force", "key": {"array": "/timestamps/force"}},
    )


def test_is_declared_never_detected(tmp_path):
    _recording(tmp_path / "rec.h5")
    assert not Hdf5Format().detect(tmp_path)


def test_frame_i_is_row_i(tmp_path):
    _recording(tmp_path / "rec.h5")
    loader = Hdf5ArrayLoader(str(tmp_path), array="/robot_state/force")
    assert len(loader) == 500 and loader.shape == (3,)
    np.testing.assert_array_equal(loader[2], [6.0, 7.0, 8.0])
    np.testing.assert_array_equal(loader[-1], [1497.0, 1498.0, 1499.0])
    with pytest.raises(IndexError):
        loader[500]


def test_clock_array_with_units(tmp_path):
    _recording(tmp_path / "rec.h5")
    fmt = Hdf5Format()
    loader = fmt.open(tmp_path, {"array": "/robot_state/pose"})
    clock = fmt.clock(loader, {"array": "/timestamps/pose", "units": ["ns"]}, "pose")
    np.testing.assert_allclose(clock[:3], [100.0, 100.1, 100.2])


def test_clock_of_another_length_is_refused(tmp_path):
    _recording(tmp_path / "rec.h5")
    fmt = Hdf5Format()
    loader = fmt.open(tmp_path, {"array": "/robot_state/pose"})
    with pytest.raises(ValueError, match="500 values for 10 frames"):
        fmt.clock(loader, {"array": "/timestamps/force"}, "pose")


def test_asynchronous_recording_loads_and_synchronizes(tmp_path):
    seq = _sequence(tmp_path)
    ds = apairo.RawDataset(seq, keys=["force", "pose"])
    assert len(ds.timestamps["force"]) == 500 and len(ds.timestamps["pose"]) == 10
    sync = ds.synchronize(reference="pose", method="nearest", tolerance=0.01)
    sample = sync[3]
    assert sample.data["pose"].shape == (7,) and sample.data["force"].shape == (3,)
    # pose 3 is at t = 100.3 s, force row 150 too (500 Hz from 100 s): ns and s
    # clocks meet on one timeline.
    assert abs(sample.timestamp - 100.3) < 1e-6
    np.testing.assert_array_equal(sample.data["force"], [450.0, 451.0, 452.0])


def test_a_root_of_recordings_is_a_dataset_of_sequences(tmp_path):
    # One file per episode: a directory per recording, one declaration at the
    # root for all of them.
    root = tmp_path / "root"
    root.mkdir()
    for i in range(3):
        _sequence(root, f"demo_{i}")
    ds = apairo.RawDataset(root, keys=["pose", "force"])
    assert len(ds.sequences) == 3
    assert len(ds.synchronize(reference="pose", method="nearest", tolerance=0.01)) == 30


def test_missing_array_field_is_named(tmp_path):
    _recording(tmp_path / "rec.h5")
    with pytest.raises(ValueError, match="array:"):
        Hdf5Format().open(tmp_path, {})


def test_unknown_array_and_scalar_are_refused(tmp_path):
    _recording(tmp_path / "rec.h5")
    with pytest.raises(KeyError, match="no array"):
        Hdf5ArrayLoader(str(tmp_path), array="/nope")
    with pytest.raises(ValueError, match="scalar"):
        Hdf5ArrayLoader(str(tmp_path), array="/meta/scalar")


def test_array_file_glob_must_match_one_file(tmp_path):
    _recording(tmp_path / "a.h5")
    _recording(tmp_path / "b.h5")
    with pytest.raises(FileNotFoundError, match="must hold one HDF5 file"):
        Hdf5ArrayLoader(str(tmp_path), array="/robot_state/force")
    with pytest.raises(FileNotFoundError, match="must match one file"):
        Hdf5ArrayLoader(str(tmp_path), array="/robot_state/force", array_file="*.h5")
    assert (
        len(
            Hdf5ArrayLoader(
                str(tmp_path), array="/robot_state/force", array_file="b.h5"
            )
        )
        == 500
    )


def test_loader_survives_pickling_for_dataloader_workers(tmp_path):
    _recording(tmp_path / "rec.h5")
    loader = Hdf5ArrayLoader(str(tmp_path), array="/robot_state/force")
    loader[0]  # opens the handle
    clone = pickle.loads(pickle.dumps(loader))
    np.testing.assert_array_equal(clone[1], loader[1])


def test_validate_flags_malformed_paths(tmp_path):
    fmt = Hdf5Format()
    assert fmt.validate("x", {"array": "/a"}, tmp_path) == []
    assert fmt.validate("x", {"array": ""}, tmp_path)
    assert fmt.validate("x", {"key": {"array": 3}}, tmp_path)


def test_declare_hints_list_the_arrays(tmp_path):
    _recording(tmp_path / "rec.h5")
    hints = "\n".join(Hdf5Format().declare_hints(tmp_path, {}))
    assert "/robot_state/force (500, 3)" in hints and "/meta/scalar" not in hints


def test_is_a_builtin_format():
    from apairo.core.formats import get_format

    assert get_format("hdf5").name == "hdf5"


def test_reassemble_declaration_ships_by_name(tmp_path):
    # The layout of a REASSEMBLE demonstration: sensors under /robot_state,
    # each clock under /timestamps, epoch seconds.
    demo = tmp_path / "2025-01-11-14-43-37"
    demo.mkdir()
    t0 = 1736603018.4
    with h5py.File(demo / "2025-01-11-14-43-37.h5", "w") as f:
        for name, width, n in (
            ("measured_force", 3, 170),
            ("measured_torque", 3, 170),
            ("joint_positions", 7, 100),
            ("pose", 7, 60),
            ("gripper_positions", 2, 100),
        ):
            f[f"robot_state/{name}"] = np.zeros((n, width))
            f[f"timestamps/{name}"] = t0 + np.arange(n) / n
        f["hama1"] = np.void(b"not a channel")
    ds = apairo.RawDataset(demo, declare="reassemble")
    assert set(ds.timestamps) == {"force", "torque", "joints", "pose", "gripper"}
    frames = ds.synchronize(reference="pose", method="nearest", tolerance=0.01)
    assert len(frames) == 60 and frames[0].data["joints"].shape == (7,)


def test_status_and_check_read_a_container_through_a_declaration(tmp_path, capsys):
    # No .apairo and no channel directory: --declare alone makes it a sequence,
    # and reading it writes nothing.
    from apairo.cli import main

    seq = tmp_path / "demo"
    seq.mkdir()
    _recording(seq / "demo.h5")
    decl = tmp_path / "decl.yaml"
    decl.write_text(yaml.safe_dump(DECLARATION))

    def run(*args):
        try:
            return main(list(args))
        except SystemExit as exc:
            return exc.code

    assert run("status", str(seq), "--declare", str(decl)) in (0, None)
    out = capsys.readouterr().out
    assert "force" in out and "hdf5" in out and "500" in out
    assert run("check", str(seq), "--declare", str(decl)) == 0
    assert not (seq / ".apairo").exists()
