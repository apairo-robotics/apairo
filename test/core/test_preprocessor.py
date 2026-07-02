"""Preprocessor callable protocol: __call__, deprecated process(), lazy transform()."""

import numpy as np
import pytest

from apairo.core.preprocessor import FramePreprocessor, SequencePreprocessor
from apairo.core.sample import Sample
from apairo.dataset.tartan_kitti import TartanKittiDataset


class _Double(FramePreprocessor):
    output_key = "doubled"
    output_loader = "npys"
    input_keys = ["velodyne_0"]
    timestamps_from = "velodyne_0"

    def __call__(self, sample: Sample) -> np.ndarray:
        return sample.data["velodyne_0"] * 2


class _Stack(SequencePreprocessor):
    output_key = "stacked"
    output_loader = "npy"
    input_keys = ["velodyne_0"]

    def __call__(self, frames) -> np.ndarray:
        return np.stack([s.data["velodyne_0"] for s in frames])


@pytest.fixture
def tartan_seq(tmp_path):
    n = 4
    vel_dir = tmp_path / "velodyne_0"
    vel_dir.mkdir()
    for i in range(n):
        np.save(vel_dir / f"{i:06d}.npy", np.full((5, 4), i, dtype=np.float32))
    np.savetxt(vel_dir / "timestamps.txt", np.arange(n, dtype=float))
    return tmp_path


# ------------------------------------------------------------- callable protocol


def test_legacy_process_warns_at_definition_and_still_works(tartan_seq):
    with pytest.warns(DeprecationWarning, match="__call__"):
        class _Legacy(FramePreprocessor):
            output_key = "legacy"
            output_loader = "npys"
            input_keys = ["velodyne_0"]
            timestamps_from = "velodyne_0"

            def process(self, sample):
                return sample.data["velodyne_0"] + 1

    p = _Legacy()
    s = Sample(data={"velodyne_0": np.zeros(3)})
    np.testing.assert_array_equal(p(s), np.ones(3))  # aliased, no warning

    # the runner path still works end-to-end
    TartanKittiDataset.run_preprocess(p, tartan_seq)
    ds = TartanKittiDataset(tartan_seq, keys=["legacy"])
    np.testing.assert_array_equal(np.asarray(ds[2].data["legacy"]), np.full((5, 4), 3))


def test_process_call_is_deprecated_delegate():
    p = _Double()
    s = Sample(data={"velodyne_0": np.ones(3)})
    with pytest.warns(DeprecationWarning, match="call the instance"):
        result = p.process(s)
    np.testing.assert_array_equal(result, np.full(3, 2.0))


def test_subclass_without_call_or_process_stays_abstract():
    class _Empty(FramePreprocessor):
        output_key = "x"
        output_loader = "npys"
        input_keys = ["a"]

    with pytest.raises(TypeError, match="abstract"):
        _Empty()


# ------------------------------------------------------- lazy preview via transform


def test_transform_runs_frame_preprocessor_lazily(tartan_seq):
    ds = TartanKittiDataset(tartan_seq, keys=["velodyne_0"])
    preview = ds.transform(_Double())

    s = preview[2]
    np.testing.assert_array_equal(np.asarray(s.data["doubled"]), np.full((5, 4), 4))
    assert "velodyne_0" in s.data                     # source untouched
    assert not (tartan_seq / "doubled").exists()      # nothing written


def test_transform_preprocessor_output_override(tartan_seq):
    ds = TartanKittiDataset(tartan_seq, keys=["velodyne_0"])
    s = ds.transform(_Double(), output="alt")[1]
    assert "alt" in s.data and "doubled" not in s.data


def test_transform_preprocessor_keep_false_drops_channel(tartan_seq):
    ds = TartanKittiDataset(tartan_seq, keys=["velodyne_0"])
    ds.transform(_Double(), keep=False)
    ds.transform(lambda s: s)  # a later step would still see it; final sample not
    assert "doubled" not in ds[0].data


def test_transform_preprocessor_missing_input_raises(tartan_seq):
    # an earlier step removes the declared input -> clear KeyError, not a
    # silent skip
    def drop_velodyne(sample):
        sample.data.pop("velodyne_0")
        return sample

    ds = TartanKittiDataset(tartan_seq, keys=["velodyne_0"])
    ds.transform(drop_velodyne).transform(_Double())
    with pytest.raises(KeyError, match="velodyne_0"):
        ds[0]


def test_transform_rejects_sequence_preprocessor(tartan_seq):
    ds = TartanKittiDataset(tartan_seq, keys=["velodyne_0"])
    with pytest.raises(TypeError, match="run_preprocess"):
        ds.transform(_Stack())


def test_transform_rejects_preprocessor_class(tartan_seq):
    ds = TartanKittiDataset(tartan_seq, keys=["velodyne_0"])
    with pytest.raises(TypeError, match="_Double\\(\\)"):
        ds.transform(_Double)


def test_same_instance_previews_then_materializes(tartan_seq):
    p = _Double()
    ds = TartanKittiDataset(tartan_seq, keys=["velodyne_0"])
    lazy = ds.transform(p)[3].data["doubled"]

    TartanKittiDataset.run_preprocess(p, tartan_seq)
    persisted = TartanKittiDataset(tartan_seq, keys=["doubled"])[3].data["doubled"]
    np.testing.assert_array_equal(np.asarray(lazy), np.asarray(persisted))
