import numpy as np
import pytest

from apairo.core import AbstractDataset, FilteredView
from apairo.core.sample import Sample
from apairo.core.utils.exceptions import KeysDuplicateError, KeysEmptyError
from test.utils.create_mock_dataset import create_mock_dataset


@pytest.fixture
def dataset():
    return create_mock_dataset()


def test_keys_setter(dataset):
    assert dataset.keys == ["key"]

    with pytest.raises(KeysEmptyError):
        dataset.keys = []

    with pytest.raises(KeysDuplicateError):
        dataset.keys = ["key_a", "key_a"]

    dataset.keys = ["key_a", "key_b"]
    assert dataset.keys == ["key_a", "key_b"]

    dataset.keys = ["key"]


def test_load(dataset):
    assert dataset.load("key", 0) == "value"


def test_len(dataset):
    assert len(dataset) == 1


def test_shape(dataset):
    assert dataset.shape == (1,)


def test_iter(dataset):
    sample = next(iter(dataset))
    assert isinstance(sample, Sample)
    assert sample.data == {"key": "value"}


def test_len_is_abstract():
    class NoLen(AbstractDataset):
        def _load(self, idx):
            return Sample(data={})

    with pytest.raises(TypeError):
        NoLen()


# ------------------------------------------------------------------ transforms


class SampleDataset(AbstractDataset):
    """Minimal dataset that returns real Sample objects for transform tests."""

    def __init__(self):
        self.keys = ["lidar", "labels"]
        self.loaders = {}

    def __len__(self):
        return 1

    def _load(self, idx):
        return Sample(
            data={
                "lidar": np.array([[1.0, 2.0], [3.0, 4.0]]),
                "labels": np.array([0, 1]),
            }
        )


@pytest.fixture
def sample_ds():
    return SampleDataset()


def test_transform_sample_level_applied(sample_ds):
    sample_ds.transform(lambda s: Sample(data={k: v * 2 for k, v in s.data.items()}))
    sample = sample_ds[0]
    np.testing.assert_array_equal(
        sample.data["lidar"], np.array([[2.0, 4.0], [6.0, 8.0]])
    )
    np.testing.assert_array_equal(sample.data["labels"], np.array([0, 2]))


def test_transform_pipeline_order(sample_ds):
    log = []
    sample_ds.transform(lambda s: (log.append("first"), s)[1])
    sample_ds.transform(lambda s: (log.append("second"), s)[1])
    sample_ds[0]
    assert log == ["first", "second"]


def test_transform_per_channel_and_sample_level_interleave(sample_ds):
    """Per-channel and sample-level steps run in registration order."""
    log = []
    sample_ds.transform("lidar", lambda arr: (log.append("channel"), arr)[1])
    sample_ds.transform(lambda s: (log.append("sample"), s)[1])
    sample_ds[0]
    assert log == ["channel", "sample"]


def test_transform_chaining_returns_self(sample_ds):
    assert sample_ds.transform(lambda s: s) is sample_ds
    assert sample_ds.transform("lidar", lambda x: x) is sample_ds


def test_transform_in_place_false_leaves_base_untouched(sample_ds):
    branch = sample_ds.transform("lidar", lambda x: x * 0, in_place=False)
    assert branch is not sample_ds
    np.testing.assert_array_equal(branch[0].data["lidar"], np.zeros((2, 2)))
    np.testing.assert_array_equal(
        sample_ds[0].data["lidar"], np.array([[1.0, 2.0], [3.0, 4.0]])
    )


def test_transform_in_place_false_branches_are_independent(sample_ds):
    v1 = sample_ds.transform("lidar", lambda x: x * 0, in_place=False)
    v2 = sample_ds.transform("lidar", lambda x: x + 10, in_place=False)
    np.testing.assert_array_equal(v1[0].data["lidar"], np.zeros((2, 2)))
    np.testing.assert_array_equal(
        v2[0].data["lidar"], np.array([[11.0, 12.0], [13.0, 14.0]])
    )


def test_transform_branch_inherits_existing_pipeline(sample_ds):
    sample_ds.transform("lidar", lambda x: x + 1)  # registered before branching
    branch = sample_ds.transform("lidar", lambda x: x * 2, in_place=False)
    np.testing.assert_array_equal(
        branch[0].data["lidar"], np.array([[4.0, 6.0], [8.0, 10.0]])
    )
    # ...and the base did not gain the branch's step.
    np.testing.assert_array_equal(
        sample_ds[0].data["lidar"], np.array([[2.0, 3.0], [4.0, 5.0]])
    )


def test_transform_in_place_false_keep_false_scoped_to_branch(sample_ds):
    branch = sample_ds.transform(
        "lidar", lambda x: x, output="_tmp", keep=False, in_place=False
    )
    assert "_tmp" not in branch[0].data
    assert not getattr(sample_ds, "_drop_keys", set())


def _double(arr):
    return arr * 2


def test_transformed_dataset_pickles(sample_ds):
    """Per-channel steps are module-level objects, not closures, so a
    transformed dataset survives pickling (spawn-based DataLoader workers)."""
    import pickle

    sample_ds.transform("lidar", _double)
    clone = pickle.loads(pickle.dumps(sample_ds))
    np.testing.assert_array_equal(
        clone[0].data["lidar"], np.array([[2.0, 4.0], [6.0, 8.0]])
    )


def test_transform_reversed_args_raises(sample_ds):
    with pytest.raises(TypeError, match="reversed"):
        sample_ds.transform(lambda x: x, "lidar")  # (fn, key) instead of (key, fn)


def test_transform_key_without_fn_raises(sample_ds):
    with pytest.raises(TypeError, match="callable"):
        sample_ds.transform("lidar")  # forgot the function


def test_transform_multi_channel_sync(sample_ds):
    def keep_first_row(s):
        s.data["lidar"] = s.data["lidar"][:1]
        s.data["labels"] = s.data["labels"][:1]
        return s

    sample_ds.transform(keep_first_row)
    sample = sample_ds[0]
    assert sample.data["lidar"].shape == (1, 2)
    assert sample.data["labels"].shape == (1,)


def test_transform_output_publishes_new_channel(sample_ds):
    sample_ds.transform("lidar", lambda x: x * 0, output="lidar_zeros")
    sample = sample_ds[0]
    assert "lidar_zeros" in sample.data
    assert "lidar" in sample.data
    np.testing.assert_array_equal(sample.data["lidar_zeros"], np.zeros((2, 2)))
    np.testing.assert_array_equal(
        sample.data["lidar"], np.array([[1.0, 2.0], [3.0, 4.0]])
    )


def test_transform_keep_false_drops_channel(sample_ds):
    sample_ds.transform("lidar", lambda x: x, output="_tmp", keep=False)
    sample = sample_ds[0]
    assert "_tmp" not in sample.data
    assert "lidar" in sample.data


def test_transform_published_channel_available_to_next_step(sample_ds):
    """A published channel is visible to subsequent pipeline steps."""
    sample_ds.transform("lidar", lambda x: x[:1], output="lidar_f")
    sample_ds.transform("labels", lambda x: x[:1], output="labels_f")
    sample_ds.transform(lambda s: s)  # reads both lidar_f and labels_f if needed
    sample = sample_ds[0]
    assert sample.data["lidar_f"].shape == (1, 2)
    assert sample.data["labels_f"].shape == (1,)


# ------------------------------------------------------------------ filter


class MultiSampleDataset(AbstractDataset):
    """Five-frame dataset: lidar row count matches the frame index + 1."""

    def __init__(self):
        self.keys = ["lidar", "labels"]
        self.loaders = {}

    def __len__(self):
        return 5

    def _load(self, idx):
        return Sample(
            data={
                "lidar": np.ones((idx + 1, 3)),  # frame i has i+1 points
                "labels": np.array([idx % 2]),  # 0 or 1
            }
        )


@pytest.fixture
def multi_ds():
    return MultiSampleDataset()


def test_filter_sample_level(multi_ds):
    view = multi_ds.filter(lambda s: s.data["lidar"].shape[0] >= 3)
    assert isinstance(view, FilteredView)
    assert len(view) == 3  # frames 2, 3, 4 (3, 4, 5 points)


def test_filter_per_channel(multi_ds):
    view = multi_ds.filter("labels", lambda lbl: lbl[0] == 1)
    assert len(view) == 2  # frames 1, 3 (odd indices)


def test_filter_returns_correct_data(multi_ds):
    view = multi_ds.filter(lambda s: s.data["lidar"].shape[0] == 1)
    assert len(view) == 1
    sample = view[0]
    assert sample.data["lidar"].shape == (1, 3)


def test_filter_empty(multi_ds):
    view = multi_ds.filter(lambda s: False)
    assert len(view) == 0


def test_filter_chaining_with_transform(multi_ds):
    view = multi_ds.filter(lambda s: s.data["lidar"].shape[0] >= 3).transform(
        "lidar", lambda x: x * 2
    )
    sample = view[0]
    np.testing.assert_array_equal(sample.data["lidar"], np.ones((3, 3)) * 2)


def test_filter_chaining_filter_on_filter(multi_ds):
    view = (
        multi_ds.filter(lambda s: s.data["lidar"].shape[0] >= 2).filter(  # frames 1-4
            lambda s: s.data["labels"][0] == 1
        )  # odd indices only
    )
    assert len(view) == 2  # frames 1 and 3


def test_filter_indices_property(multi_ds):
    view = multi_ds.filter(lambda s: s.data["lidar"].shape[0] >= 3)
    idx = view.indices
    assert isinstance(idx, np.ndarray)
    np.testing.assert_array_equal(idx, [2, 3, 4])


def test_filter_from_precomputed_indices(multi_ds):
    indices = np.array([0, 2, 4], dtype=np.int64)
    view = multi_ds.filter(indices)
    assert len(view) == 3
    assert view[0].data["lidar"].shape == (1, 3)  # frame 0 has 1 point
    assert view[1].data["lidar"].shape == (3, 3)  # frame 2 has 3 points
    assert view[2].data["lidar"].shape == (5, 3)  # frame 4 has 5 points


def test_filter_parent_transforms_applied(multi_ds):
    """Transforms registered on the parent must be visible in the filtered view."""
    multi_ds.transform("lidar", lambda x: x * 10)
    view = multi_ds.filter(lambda s: s.data["lidar"].shape[0] >= 3)
    sample = view[0]
    np.testing.assert_array_equal(sample.data["lidar"], np.ones((3, 3)) * 10)


def test_nested_iteration_is_independent(multi_ds):
    """Two simultaneous iterations over the same dataset must not interfere."""
    pairs = [(0, 0) for _ in multi_ds for _ in multi_ds]
    assert len(pairs) == len(multi_ds) ** 2


def test_per_channel_filter_rejects_async():
    class AsyncDS(AbstractDataset):
        def __init__(self):
            self.keys = ["a"]
            self.timestamps = {"a": np.array([0.0, 1.0])}

        def __len__(self):
            return 2

        def _load(self, idx):
            return Sample(data={"a": np.zeros(1)}, timestamp=float(idx))

    with pytest.raises(ValueError, match="synchronize"):
        AsyncDS().filter("a", lambda x: True)


def test_view_is_synchronous_delegates_to_parent():
    class AsyncDS(AbstractDataset):
        def __init__(self):
            self.keys = ["a"]
            self.timestamps = {"a": np.array([0.0, 1.0])}

        def __len__(self):
            return 2

        def _load(self, idx):
            return Sample(data={"a": np.zeros(1)}, timestamp=float(idx))

    ds = AsyncDS()
    assert not ds.filter(np.array([0])).is_synchronous
    assert not ds.select(["a"]).is_synchronous
    assert not ds.cache().is_synchronous


def test_filter_indices_roundtrip(tmp_path, multi_ds):
    view = multi_ds.filter(lambda s: s.data["lidar"].shape[0] >= 3)
    path = tmp_path / "indices.npy"
    np.save(path, view.indices)

    view2 = multi_ds.filter(np.load(path))
    assert len(view2) == len(view)
    for i in range(len(view)):
        np.testing.assert_array_equal(view[i].data["lidar"], view2[i].data["lidar"])
