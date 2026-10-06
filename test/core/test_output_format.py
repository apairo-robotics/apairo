"""Where a preprocess output is written, and in which format.

apairo does not impose a storage: an output goes in the format the run asks
for (``output_format=``), else the one its preprocessor declares
(``output_loader``), else the format of its input channel when that format can
hold it -- an image mask stays an image -- else ``npys`` / ``npy``. Any format
that writes is a target, a plugin's included, and a format conversion is just
a preprocess (:class:`~apairo.preprocess.Convert`). Whatever is written reads
back unchanged, in the asynchronous family as in a profiled dataset.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from apairo.core.config import read_config, write_config
from apairo.core.preprocessor import FramePreprocessor, SequencePreprocessor
from apairo.core.sample import Sample
from apairo.dataset.raw import RawDataset
from apairo.preprocess import Convert
from test.dataset.test_profiled_dataset import _GooseDS, goose_root  # noqa: F401

N = 4
CLOCK = 100.0 + np.arange(N) * 0.1


def _images(seq, name="cam"):
    d = seq / name
    d.mkdir(parents=True)
    for i in range(N):
        Image.fromarray(np.full((6, 8, 3), 10 * i, np.uint8)).save(d / f"{i:06d}.png")
    np.savetxt(d / "timestamps.txt", CLOCK)
    write_config(
        seq, {"version": 1, "channels": {name: {"kind": "raw", "loader": "img"}}}
    )
    return seq


def _clouds(seq, name="lidar"):
    d = seq / name
    d.mkdir(parents=True)
    for i in range(N):
        np.save(d / f"{i:06d}.npy", np.full((5, 3), i, np.float32))
    np.savetxt(d / "timestamps.txt", CLOCK)
    write_config(
        seq, {"version": 1, "channels": {name: {"kind": "raw", "loader": "npys"}}}
    )
    return seq


class _Mask(FramePreprocessor):  # no output_loader: apairo chooses
    output_key = "mask"
    input_keys = ["cam"]

    def __call__(self, sample: Sample) -> np.ndarray:
        return (sample.data["cam"][..., 0] > 15).astype(np.uint8) * 255


class _Depth(FramePreprocessor):  # float: an image cannot hold it
    output_key = "depth"
    input_keys = ["cam"]

    def __call__(self, sample: Sample) -> np.ndarray:
        return sample.data["cam"][..., 0].astype(np.float32) / 7


class _Scale(FramePreprocessor):
    output_key = "scaled"
    input_keys = ["lidar"]

    def __call__(self, sample: Sample) -> np.ndarray:
        return sample.data["lidar"] * 2


def _loader(seq, key):
    return read_config(seq)["channels"][key]["loader"]


# ─────────────────────────── the format is chosen, not imposed ────────────────


def test_an_output_stays_in_its_input_format_when_it_fits(tmp_path):
    seq = _images(tmp_path / "seq")
    RawDataset.run_preprocess(_Mask(), seq)
    assert _loader(seq, "mask") == "img"
    assert sorted(p.name for p in (seq / "mask").glob("*.png")) == [
        f"{i:06d}.png" for i in range(N)
    ]
    ds = RawDataset(seq, keys=["mask"])
    np.testing.assert_array_equal(ds.loaders["mask"][2], np.full((6, 8), 255, np.uint8))
    np.testing.assert_allclose(ds.timestamps["mask"], CLOCK)


def test_an_output_the_input_format_cannot_hold_falls_back_to_npys(tmp_path):
    seq = _images(tmp_path / "seq")
    RawDataset.run_preprocess(_Depth(), seq)
    assert _loader(seq, "depth") == "npys"
    ds = RawDataset(seq, keys=["depth"])
    assert ds.loaders["depth"][1].dtype == np.float32


def test_a_declared_output_loader_wins_over_the_input_format(tmp_path):
    class _NpysMask(_Mask):
        output_loader = "npys"

    seq = _images(tmp_path / "seq")
    RawDataset.run_preprocess(_NpysMask(), seq)
    assert _loader(seq, "mask") == "npys"


def test_the_run_chooses_any_format_that_writes(tmp_path):
    pytest.importorskip("zarr")
    seq = _clouds(tmp_path / "seq")
    RawDataset.run_preprocess(_Scale(), seq, output_format="zarr")
    assert _loader(seq, "scaled") == "zarr"
    assert (seq / "scaled" / "timestamps.txt").is_file()
    ds = RawDataset(seq, keys=["scaled"])
    assert len(ds.loaders["scaled"]) == N
    np.testing.assert_array_equal(
        ds.loaders["scaled"][3], np.full((5, 3), 6, np.float32)
    )
    np.testing.assert_allclose(ds.timestamps["scaled"], CLOCK)


def test_a_read_only_format_is_refused_by_name(tmp_path):
    seq = _clouds(tmp_path / "seq")
    with pytest.raises(
        ValueError, match="'pcd' format is read-only. Formats that write"
    ):
        RawDataset.run_preprocess(_Scale(), seq, output_format="pcd")
    assert not (seq / "scaled").exists()


def test_a_format_that_cannot_hold_the_output_is_refused(tmp_path):
    seq = _clouds(tmp_path / "seq")
    with pytest.raises(ValueError, match="cannot be stored in the 'bin' format"):
        RawDataset.run_preprocess(_Scale(), seq, output_format="bin")


def test_a_ragged_output_is_not_stacked(tmp_path):
    class _Ragged(_Scale):
        def __call__(self, sample):
            return np.zeros((int(sample.data["lidar"][0, 0]) + 1, 3))

    seq = _clouds(tmp_path / "seq")
    with pytest.raises(ValueError, match="changes shape from frame to frame"):
        RawDataset.run_preprocess(_Ragged(), seq, output_format="npy")


def test_a_sequence_preprocessor_follows_the_same_rule(tmp_path):
    class _Centroids(SequencePreprocessor):
        output_key = "centroid"
        input_keys = ["lidar"]

        def __call__(self, frames):
            return np.stack([s.data["lidar"].mean(axis=0) for s in frames])

    seq = _clouds(tmp_path / "seq")
    RawDataset.run_preprocess(_Centroids(), seq)  # npys input: one file per frame
    assert _loader(seq, "centroid") == "npys"
    assert len(list((seq / "centroid").glob("*.npy"))) == N

    class _Stacked(_Centroids):
        output_key = "centroid_stacked"
        output_loader = "npy"

    RawDataset.run_preprocess(_Stacked(), seq)
    assert (seq / "centroid_stacked" / "centroid_stacked.npy").is_file()
    ds = RawDataset(seq, keys=["centroid_stacked"])
    np.testing.assert_allclose(ds.loaders["centroid_stacked"][2], [2.0, 2.0, 2.0])


# ──────────────────────────── a conversion is a preprocess ────────────────────


def test_convert_copies_a_channel_into_another_format(tmp_path):
    pytest.importorskip("zarr")
    seq = _clouds(tmp_path / "seq")
    RawDataset.run_preprocess(Convert("lidar", to="zarr"), seq)
    entry = read_config(seq)["channels"]["lidar_zarr"]
    assert entry["loader"] == "zarr"
    assert entry["kind"] == "preprocess"
    assert entry["sources"] == ["lidar"]
    ds = RawDataset(seq, keys=["lidar", "lidar_zarr"])
    for i in range(N):
        np.testing.assert_array_equal(
            ds.loaders["lidar_zarr"][i], ds.loaders["lidar"][i]
        )
    np.testing.assert_allclose(ds.timestamps["lidar_zarr"], ds.timestamps["lidar"])


def test_convert_a_table_into_a_stacked_array(tmp_path):
    seq = tmp_path / "seq"
    (seq / "imu").mkdir(parents=True)
    (seq / "imu" / "data.csv").write_text(
        "t,ax,ay\n" + "".join(f"{100 + k * 0.01:.2f},{k}.5,{-k}.25\n" for k in range(6))
    )
    write_config(
        seq,
        {
            "version": 1,
            "channels": {
                "imu": {"kind": "raw", "loader": "csv", "key": {"column": "t"}}
            },
        },
    )
    RawDataset.run_preprocess(Convert("imu", to="npy", output_key="imu_array"), seq)
    ds = RawDataset(seq, keys=["imu", "imu_array"])
    np.testing.assert_array_equal(ds.loaders["imu_array"][4], [4.5, -4.25])
    np.testing.assert_allclose(ds.timestamps["imu_array"], ds.timestamps["imu"])


def test_convert_refuses_a_target_that_cannot_hold_the_frames(tmp_path):
    seq = _clouds(tmp_path / "seq")
    with pytest.raises(ValueError, match="cannot be stored in the 'img' format"):
        RawDataset.run_preprocess(Convert("lidar", to="img"), seq)


# ───────────────── a profiled dataset reads back what was written ─────────────


class _GooseMask(FramePreprocessor):
    output_key = "ground"
    input_keys = ["lidar"]

    def __call__(self, sample: Sample) -> np.ndarray:
        return (sample.data["lidar"][:, :1] > 0).astype(np.uint8) * 255


@pytest.mark.parametrize("output_format", ["img", "npys", "zarr", "npy"])
def test_a_profiled_dataset_reads_any_written_format(goose_root, output_format):  # noqa: F811
    if output_format == "zarr":
        pytest.importorskip("zarr")
    _GooseDS.run_preprocess(_GooseMask(), goose_root, output_format=output_format)
    ds = _GooseDS(goose_root, keys=["lidar", "ground"])
    assert len(ds) == 6
    for i in range(len(ds)):
        expected = _GooseMask()(Sample(data={"lidar": ds.loaders["lidar"][i]}))
        np.testing.assert_array_equal(ds.loaders["ground"][i], expected)
