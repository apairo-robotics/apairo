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
    assert "velodyne_0" in s.data  # source untouched
    assert not (tartan_seq / "doubled").exists()  # nothing written


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


# ------------------------------------------------------------------ multi-output


class _Split(FramePreprocessor):
    output_keys = ["halved", "negated"]
    output_loader = "npys"
    input_keys = ["velodyne_0"]
    timestamps_from = "velodyne_0"
    sources = ["velodyne_0"]

    def __call__(self, sample: Sample) -> dict[str, np.ndarray]:
        pts = sample.data["velodyne_0"]
        return {"halved": pts / 2, "negated": -pts}


class _SplitStacked(SequencePreprocessor):
    output_keys = ["means", "maxes"]
    output_loader = "npy"
    input_keys = ["velodyne_0"]
    timestamps_from = "velodyne_0"

    def __call__(self, frames) -> dict[str, np.ndarray]:
        pts = [s.data["velodyne_0"] for s in frames]
        return {
            "means": np.stack([p.mean(axis=0) for p in pts]),
            "maxes": np.stack([p.max(axis=0) for p in pts]),
        }


def test_output_key_and_output_keys_are_exclusive():
    with pytest.raises(TypeError, match="exclusive"):

        class _Both(FramePreprocessor):
            output_key = "a"
            output_keys = ["a", "b"]
            output_loader = "npys"
            input_keys = ["velodyne_0"]

            def __call__(self, sample):
                return {}


def test_output_keys_must_be_unique_and_nonempty():
    for bad in ([], ["a", "a"]):
        with pytest.raises(TypeError, match="unique"):

            class _Bad(FramePreprocessor):
                output_keys = bad
                output_loader = "npys"
                input_keys = ["velodyne_0"]

                def __call__(self, sample):
                    return {}


def test_multi_output_run_preprocess_registers_every_key(tartan_seq):
    from apairo.core.config import read_config

    TartanKittiDataset.run_preprocess(_Split(), tartan_seq)

    channels = read_config(tartan_seq)["channels"]
    for key in ("halved", "negated"):
        assert channels[key]["kind"] == "preprocess"
        assert channels[key]["timestamps_from"] == "velodyne_0"
        assert channels[key]["sources"] == ["velodyne_0"]
        assert (tartan_seq / key / "timestamps.txt").exists()

    # each channel is individually selectable
    src = TartanKittiDataset(tartan_seq, keys=["velodyne_0"])[2].data["velodyne_0"]
    halved = TartanKittiDataset(tartan_seq, keys=["halved"])[2].data["halved"]
    negated = TartanKittiDataset(tartan_seq, keys=["negated"])[2].data["negated"]
    np.testing.assert_array_equal(np.asarray(halved), np.asarray(src) / 2)
    np.testing.assert_array_equal(np.asarray(negated), -np.asarray(src))


def test_multi_output_stacked_sequence_writes_one_file_per_key(tartan_seq):
    TartanKittiDataset.run_preprocess(_SplitStacked(), tartan_seq)

    for key in ("means", "maxes"):
        assert (tartan_seq / key / f"{key}.npy").exists()
    means = np.load(tartan_seq / "means" / "means.npy")
    assert means.shape == (4, 4)


def test_multi_output_wrong_keys_raises(tartan_seq):
    class _Liar(FramePreprocessor):
        output_keys = ["a", "b"]
        output_loader = "npys"
        input_keys = ["velodyne_0"]
        timestamps_from = "velodyne_0"

        def __call__(self, sample):
            return {"a": sample.data["velodyne_0"]}  # missing "b"

    with pytest.raises(ValueError, match="exactly those keys"):
        TartanKittiDataset.run_preprocess(_Liar(), tartan_seq)


def test_multi_output_overwrite_check_covers_every_key(tartan_seq):
    # only the *second* key pre-exists: the run must still refuse
    (tartan_seq / "negated").mkdir()
    np.save(tartan_seq / "negated" / "000000.npy", np.zeros(3))

    with pytest.raises(FileExistsError, match="negated"):
        TartanKittiDataset.run_preprocess(_Split(), tartan_seq)


def test_transform_multi_output_publishes_every_key(tartan_seq):
    ds = TartanKittiDataset(tartan_seq, keys=["velodyne_0"])
    s = ds.transform(_Split())[1]
    np.testing.assert_array_equal(
        np.asarray(s.data["halved"]), np.asarray(s.data["velodyne_0"]) / 2
    )
    assert "negated" in s.data
    assert not (tartan_seq / "halved").exists()  # nothing written


def test_transform_multi_output_rejects_output_override(tartan_seq):
    ds = TartanKittiDataset(tartan_seq, keys=["velodyne_0"])
    with pytest.raises(TypeError, match="output_keys"):
        ds.transform(_Split(), output="alt")


def test_transform_multi_output_keep_false_drops_every_key(tartan_seq):
    ds = TartanKittiDataset(tartan_seq, keys=["velodyne_0"])
    ds.transform(_Split(), keep=False)
    s = ds[0]
    assert "halved" not in s.data and "negated" not in s.data
