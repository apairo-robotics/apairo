"""Integration tests: async derived channels are self-contained (own timestamps.txt)."""

import numpy as np
import pytest

from apairo.dataset.tartan_kitti import TartanKittiDataset
from apairo.core.preprocessor import FramePreprocessor
from apairo.core.sample import Sample


class _Identity(FramePreprocessor):
    output_key = "voxel"
    output_loader = "npys"
    input_keys = ["velodyne_0"]
    timestamps_from = "velodyne_0"
    sources = ["velodyne_0"]

    def process(self, sample: Sample) -> np.ndarray:
        return sample.data["velodyne_0"]


@pytest.fixture
def tartan_seq(tmp_path):
    n = 4
    vel_dir = tmp_path / "velodyne_0"
    vel_dir.mkdir()
    for i in range(n):
        np.save(vel_dir / f"{i:06d}.npy", np.random.rand(10, 4).astype(np.float32))
    np.savetxt(vel_dir / "timestamps.txt", np.arange(n, dtype=float))
    return tmp_path


def test_runner_writes_own_timestamps(tartan_seq):
    TartanKittiDataset.run_preprocess(_Identity(), tartan_seq)
    assert (tartan_seq / "voxel" / "timestamps.txt").exists()


def test_derived_loads_without_source_in_keys(tartan_seq):
    TartanKittiDataset.run_preprocess(_Identity(), tartan_seq)
    ds = TartanKittiDataset(tartan_seq, keys=["voxel"])
    assert len(ds) == 4
    s = ds[0]
    assert "voxel" in s.data
    assert "velodyne_0" not in s.data
    assert s.timestamp is not None


def test_derived_timestamps_match_source(tartan_seq):
    TartanKittiDataset.run_preprocess(_Identity(), tartan_seq)
    ds_derived = TartanKittiDataset(tartan_seq, keys=["voxel"])
    ds_source = TartanKittiDataset(tartan_seq, keys=["velodyne_0"])
    np.testing.assert_array_equal(
        ds_derived.timestamps["voxel"],
        ds_source.timestamps["velodyne_0"],
    )
