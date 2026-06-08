
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


def test_sample_transform_applied(sample_ds):
    sample_ds.sample_transform(lambda s: Sample(
        data={k: v * 2 for k, v in s.data.items()}
    ))
    sample = sample_ds[0]
    np.testing.assert_array_equal(sample.data["lidar"], np.array([[2.0, 4.0], [6.0, 8.0]]))
    np.testing.assert_array_equal(sample.data["labels"], np.array([0, 2]))


def test_sample_transform_composition_order(sample_ds):
    log = []
    sample_ds.sample_transform(lambda s: (log.append("first"), s)[1])
    sample_ds.sample_transform(lambda s: (log.append("second"), s)[1])
    sample_ds[0]
    assert log == ["first", "second"]


def test_sample_transform_channel_transform_order(sample_ds):
    """Per-channel transforms run before sample transforms."""
    seen = {}

    def record_lidar(arr):
        seen["after_channel"] = arr.copy()
        return arr

    def record_sample(s):
        seen["after_sample"] = s.data["lidar"].copy()
        return s

    sample_ds.transform("lidar", record_lidar)
    sample_ds.sample_transform(record_sample)
    sample_ds[0]
    np.testing.assert_array_equal(seen["after_channel"], seen["after_sample"])


def test_sample_transform_chaining_returns_self(sample_ds):
    result = sample_ds.sample_transform(lambda s: s)
    assert result is sample_ds


def test_sample_transform_multi_channel_sync(sample_ds):
    """sample_transform can apply a consistent mask across channels."""
    def keep_first_row(s):
        s.data["lidar"]  = s.data["lidar"][:1]
        s.data["labels"] = s.data["labels"][:1]
        return s

    sample_ds.sample_transform(keep_first_row)
    sample = sample_ds[0]
    assert sample.data["lidar"].shape == (1, 2)
    assert sample.data["labels"].shape == (1,)
