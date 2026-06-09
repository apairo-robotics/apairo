
import numpy as np
import pytest
from test.utils.create_mock_dataset import create_mock_dataset
from apairo.core import AbstractDataset, FilteredView
from apairo.core.utils.exceptions import KeysEmptyWarning, KeysDuplicateWarning
from apairo.core.sample import Sample


@pytest.fixture
def dataset():
    return create_mock_dataset()


def test_keys_setter(dataset):
    assert dataset.keys == ["key"]

    with pytest.raises(KeysEmptyWarning):  # Assuming Exception type from logic
        dataset.keys = []

    with pytest.raises(KeysDuplicateWarning):  # Assuming Exception type
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
    # Depending on implementation, iter might return a dict or tuple
    # Original test expected {"key": ["value"]}
    assert next(iter(dataset)) == {"key": ["value"]}


# ------------------------------------------------------------------ transforms

class SampleDataset(AbstractDataset):
    """Minimal dataset that returns real Sample objects for transform tests."""

    def __init__(self):
        self.keys = ["lidar", "labels"]
        self.loaders = {}

    def __len__(self):
        return 1

    def _load(self, idx):
        return Sample(data={
            "lidar":  np.array([[1.0, 2.0], [3.0, 4.0]]),
            "labels": np.array([0, 1]),
        })


@pytest.fixture
def sample_ds():
    return SampleDataset()


def test_transform_sample_level_applied(sample_ds):
    sample_ds.transform(lambda s: Sample(data={k: v * 2 for k, v in s.data.items()}))
    sample = sample_ds[0]
    np.testing.assert_array_equal(sample.data["lidar"], np.array([[2.0, 4.0], [6.0, 8.0]]))
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


def test_transform_multi_channel_sync(sample_ds):
    def keep_first_row(s):
        s.data["lidar"]  = s.data["lidar"][:1]
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
    np.testing.assert_array_equal(sample.data["lidar"], np.array([[1.0, 2.0], [3.0, 4.0]]))


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
        return Sample(data={
            "lidar":  np.ones((idx + 1, 3)),   # frame i has i+1 points
            "labels": np.array([idx % 2]),      # 0 or 1
        })


@pytest.fixture
def multi_ds():
    return MultiSampleDataset()


def test_filter_sample_level(multi_ds):
    view = multi_ds.filter(lambda s: s.data["lidar"].shape[0] >= 3)
    assert isinstance(view, FilteredView)
    assert len(view) == 3   # frames 2, 3, 4 (3, 4, 5 points)


def test_filter_per_channel(multi_ds):
    view = multi_ds.filter("labels", lambda lbl: lbl[0] == 1)
    assert len(view) == 2   # frames 1, 3 (odd indices)


def test_filter_returns_correct_data(multi_ds):
    view = multi_ds.filter(lambda s: s.data["lidar"].shape[0] == 1)
    assert len(view) == 1
    sample = view[0]
    assert sample.data["lidar"].shape == (1, 3)


def test_filter_empty(multi_ds):
    view = multi_ds.filter(lambda s: False)
    assert len(view) == 0


def test_filter_chaining_with_transform(multi_ds):
    view = (
        multi_ds
        .filter(lambda s: s.data["lidar"].shape[0] >= 3)
        .transform("lidar", lambda x: x * 2)
    )
    sample = view[0]
    np.testing.assert_array_equal(sample.data["lidar"], np.ones((3, 3)) * 2)


def test_filter_chaining_filter_on_filter(multi_ds):
    view = (
        multi_ds
        .filter(lambda s: s.data["lidar"].shape[0] >= 2)   # frames 1-4
        .filter(lambda s: s.data["labels"][0] == 1)         # odd indices only
    )
    assert len(view) == 2   # frames 1 and 3


def test_filter_indices_property(multi_ds):
    view = multi_ds.filter(lambda s: s.data["lidar"].shape[0] >= 3)
    idx = view.indices
    assert isinstance(idx, np.ndarray)
    np.testing.assert_array_equal(idx, [2, 3, 4])


def test_filter_from_precomputed_indices(multi_ds):
    indices = np.array([0, 2, 4], dtype=np.int64)
    view = multi_ds.filter(indices)
    assert len(view) == 3
    assert view[0].data["lidar"].shape == (1, 3)   # frame 0 has 1 point
    assert view[1].data["lidar"].shape == (3, 3)   # frame 2 has 3 points
    assert view[2].data["lidar"].shape == (5, 3)   # frame 4 has 5 points


def test_filter_indices_roundtrip(tmp_path, multi_ds):
    view = multi_ds.filter(lambda s: s.data["lidar"].shape[0] >= 3)
    path = tmp_path / "indices.npy"
    np.save(path, view.indices)

    view2 = multi_ds.filter(np.load(path))
    assert len(view2) == len(view)
    for i in range(len(view)):
        np.testing.assert_array_equal(view[i].data["lidar"], view2[i].data["lidar"])
