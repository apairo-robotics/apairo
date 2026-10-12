"""A preprocess run never corrupts what is already there.

An output may replace an earlier output of its own name -- wholly, so files of
a previous format are never read back with the new ones -- and nothing else: not
a raw channel, a sub-channel, another channel's public name, nor data apairo
did not write. Every output's format is decided before any is written, and a
stacked format is not chosen where it could not hold the run's frames.
"""

from __future__ import annotations

import numpy as np
import pytest

from apairo.core.config import read_config, write_config
from apairo.core.preprocessor import FramePreprocessor
from apairo.core.sample import Sample
from apairo.dataset.raw import RawDataset
from apairo.preprocess import Convert

N = 4


def _seq(path, *, suffix=False, **extra):
    d = path / "lidar"
    d.mkdir(parents=True)
    for i in range(N):
        np.save(d / f"{i:06d}.npy", np.full((3, 3), i, np.float32))
        if suffix:
            np.save(d / f"{i:06d}_intensity.npy", np.full(3, i, np.uint8))
    np.savetxt(d / "timestamps.txt", 100 + np.arange(N) * 0.1)
    channels = {"lidar": {"kind": "raw", "loader": "npys", **extra}}
    if suffix:
        channels["lidar_intensity"] = {
            "kind": "raw",
            "loader": "npys",
            "directory": "lidar",
            "suffix": "intensity",
        }
    write_config(path, {"version": 1, "channels": channels})
    return path


def test_overwrite_replaces_the_output_wholly(tmp_path):
    seq = _seq(tmp_path / "seq")
    RawDataset.run_preprocess(Convert("lidar", to="npys", output_key="copy"), seq)
    RawDataset.run_preprocess(
        Convert("lidar", to="npy", output_key="copy"), seq, overwrite=True
    )
    assert sorted(p.name for p in (seq / "copy").iterdir()) == [
        "copy.npy",
        "timestamps.txt",
    ]
    ds = RawDataset(seq, keys=["lidar", "copy"])
    for i in range(N):
        np.testing.assert_array_equal(ds.loaders["copy"][i], ds.loaders["lidar"][i])


@pytest.mark.parametrize("name", ["lidar", "lidar_intensity"])
def test_an_output_never_replaces_a_raw_channel(tmp_path, name):
    seq = _seq(tmp_path / "seq", suffix=True)
    with pytest.raises(FileExistsError, match=f"'{name}' is a raw channel"):
        RawDataset.run_preprocess(Convert("lidar", to="npy", output_key=name), seq)
    assert read_config(seq)["channels"][name]["kind"] == "raw"


def test_an_output_never_takes_a_channel_s_public_name(tmp_path):
    seq = _seq(tmp_path / "seq")
    (seq / "apairo.yaml").write_text(
        "version: 1\nchannels:\n  lidar: {alias: points}\n"
    )
    with pytest.raises(
        FileExistsError, match="'points' is the public name of channel 'lidar'"
    ):
        RawDataset.run_preprocess(Convert("points", to="npy", output_key="points"), seq)
    RawDataset(seq)  # still loads


def test_every_sequence_of_a_root_is_checked(tmp_path):
    root = tmp_path / "root"
    _seq(root / "a")
    _seq(root / "b")
    depth = root / "b" / "depth"
    depth.mkdir()
    for i in range(N):
        np.save(depth / f"{i:06d}.npy", np.zeros(2))
    np.savetxt(depth / "timestamps.txt", 100 + np.arange(N) * 0.1)
    cfg = read_config(root / "b")
    cfg["channels"]["depth"] = {"kind": "raw", "loader": "npys"}
    write_config(root / "b", cfg)
    with pytest.raises(FileExistsError, match="'depth' is a raw channel in .*b"):
        RawDataset.run_preprocess(Convert("lidar", to="npys", output_key="depth"), root)
    assert sorted(p.name for p in depth.iterdir())[0] == "000000.npy"


def test_data_apairo_did_not_write_is_left_alone(tmp_path):
    seq = _seq(tmp_path / "seq")
    (seq / "notes").mkdir()
    (seq / "notes" / "field_log.txt").write_text("kept by hand")
    with pytest.raises(FileExistsError, match="holds data apairo did not write"):
        RawDataset.run_preprocess(
            Convert("lidar", to="npys", output_key="notes"), seq, overwrite=True
        )
    assert (seq / "notes" / "field_log.txt").read_text() == "kept by hand"


def test_a_refused_output_leaves_no_other_half_written(tmp_path):
    class Two(FramePreprocessor):
        output_keys = ["a", "b"]
        input_keys = ["lidar"]
        output_loader = "bin"

        def __call__(self, sample: Sample):
            return {"a": np.zeros((5, 4), np.float32), "b": np.zeros(3)}

    seq = _seq(tmp_path / "seq")
    with pytest.raises(ValueError, match="output 'b'"):
        RawDataset.run_preprocess(Two(), seq)
    assert not (seq / "a").exists()


class _Crop(FramePreprocessor):
    output_key = "crop"
    input_keys = ["imu"]

    def __call__(self, sample: Sample):
        rows = sample.data["imu"]
        return rows[: 1 + int(rows[0, 0]) // 6 % 3]  # frame i: 1 + i % 3 rows


def _stacked_input(path):
    d = path / "imu"
    d.mkdir(parents=True)
    np.save(d / "imu.npy", np.arange(N * 3 * 2, dtype=np.float64).reshape(N, 3, 2))
    np.savetxt(d / "timestamps.txt", 100 + np.arange(N) * 0.1)
    write_config(
        path, {"version": 1, "channels": {"imu": {"kind": "raw", "loader": "npy"}}}
    )
    return path


def test_a_stacked_format_chosen_for_its_input_falls_back_when_ragged(tmp_path):
    seq = _stacked_input(tmp_path / "seq")
    RawDataset.run_preprocess(_Crop(), seq)
    assert read_config(seq)["channels"]["crop"]["loader"] == "npys"
    ds = RawDataset(seq, keys=["crop"])
    assert {ds.loaders["crop"][i].shape[0] for i in range(N)} == {1, 2, 3}


def test_a_stacked_format_asked_for_still_refuses_ragged_output(tmp_path):
    seq = _stacked_input(tmp_path / "seq")
    with pytest.raises(ValueError, match="changes shape from frame to frame"):
        RawDataset.run_preprocess(_Crop(), seq, output_format="npy")
    assert not (seq / "crop").exists()


def test_a_stacked_format_is_not_used_for_a_split_run(tmp_path):
    """A split listing frames selects some of each sequence's: a stacked array,
    row i being frame i, could not hold them."""
    import shutil
    from pathlib import Path

    from apairo.dataset import Rellis3DDataset

    root = tmp_path / "rellis"
    shutil.copytree(Path(__file__).parent.parent / "assets" / "mini_rellis", root)

    class Mean(FramePreprocessor):
        output_key = "mean_height"
        input_keys = ["lidar"]

        def __call__(self, sample: Sample):
            return sample.data["lidar"][:, 2].mean(keepdims=True)

    with pytest.raises(ValueError, match="covers a selection of them"):
        Rellis3DDataset.run_preprocess(Mean(), root, split="val", output_format="npy")
    Rellis3DDataset.run_preprocess(Mean(), root, split="val")  # per frame
    val = Rellis3DDataset(root, keys=["lidar", "mean_height"], split="val")
    for i in range(len(val)):
        np.testing.assert_allclose(
            val.loaders["mean_height"][i],
            val.loaders["lidar"][i][:, 2].mean(keepdims=True),
        )
