"""run_preprocess(..., reuse=True): recipe-addressed idempotency."""

import numpy as np
import pytest

from apairo.core.config import read_config
from apairo.core.preprocessor import FramePreprocessor
from apairo.core.sample import Sample
from apairo.dataset.tartan_kitti import TartanKittiDataset
from apairo.preprocess.runner import _recipe_key


class _Scale(FramePreprocessor):
    output_key = "scaled"
    output_loader = "npys"
    input_keys = ["velodyne_0"]
    timestamps_from = "velodyne_0"

    def __init__(self, factor: float) -> None:
        self.factor = factor

    def __call__(self, sample: Sample) -> np.ndarray:
        return sample.data["velodyne_0"] * self.factor


@pytest.fixture
def seq(tmp_path):
    vel = tmp_path / "velodyne_0"
    vel.mkdir()
    for i in range(4):
        np.save(vel / f"{i:06d}.npy", np.full((5, 4), i, dtype=np.float32))
    np.savetxt(vel / "timestamps.txt", np.arange(4, dtype=float))
    return tmp_path


def _scaled(seq, i):
    return np.load(seq / "scaled" / f"{i:06d}.npy")


def test_recipe_is_recorded(seq):
    TartanKittiDataset.run_preprocess(_Scale(2), seq)
    assert read_config(seq)["channels"]["scaled"].get("recipe")


def test_recipe_depends_on_declared_params():
    assert _recipe_key(_Scale(2)) != _recipe_key(_Scale(3))
    assert _recipe_key(_Scale(2)) == _recipe_key(_Scale(2))


def test_reuse_skips_matching_recipe(seq):
    TartanKittiDataset.run_preprocess(_Scale(2), seq)
    before = _scaled(seq, 1)
    np.testing.assert_array_equal(before, np.full((5, 4), 2, dtype=np.float32))
    # a matching recipe is a no-op: no raise, output untouched.
    TartanKittiDataset.run_preprocess(_Scale(2), seq, reuse=True)
    np.testing.assert_array_equal(_scaled(seq, 1), before)


def test_reuse_regenerates_changed_recipe(seq):
    TartanKittiDataset.run_preprocess(_Scale(2), seq)
    # a different scalar param -> different recipe -> regenerate, no overwrite needed.
    TartanKittiDataset.run_preprocess(_Scale(3), seq, reuse=True)
    np.testing.assert_array_equal(_scaled(seq, 1), np.full((5, 4), 3, dtype=np.float32))
    assert read_config(seq)["channels"]["scaled"]["recipe"] == _recipe_key(_Scale(3))


def test_default_still_raises_on_existing(seq):
    TartanKittiDataset.run_preprocess(_Scale(2), seq)
    with pytest.raises(FileExistsError):
        TartanKittiDataset.run_preprocess(_Scale(2), seq)  # no reuse, no overwrite
