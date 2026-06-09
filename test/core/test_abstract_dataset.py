
import numpy as np
import pytest
from test.utils.create_mock_dataset import create_mock_dataset
from apairo.core import AbstractDataset
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
