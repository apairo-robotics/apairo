import numpy as np
import pytest
from apairo.core import AbstractDataset
from apairo.core.sample import Sample
from apairo.dataset.zip import ZipDataset


class _DS(AbstractDataset):
    """Minimal dataset with configurable keys and per-frame values."""

    def __init__(self, n: int, keys: list[str], value_fn=None):
        self._keys = keys
        self._n = n
        self._value_fn = value_fn or (lambda key, idx: np.full(3, idx, dtype=float))

    def __len__(self):
        return self._n

    def _load(self, idx):
        return Sample(data={k: self._value_fn(k, idx) for k in self._keys})


@pytest.fixture
def ds_a():
    return _DS(5, ["lidar", "labels"])

@pytest.fixture
def ds_b():
    return _DS(5, ["trav_gt"])

@pytest.fixture
def ds_c():
    return _DS(5, ["ground_height"])


# ------------------------------------------------------------------ construction

def test_basic_merge(ds_a, ds_b):
    z = ZipDataset(ds_a, ds_b)
    assert len(z) == 5
    assert set(z._keys) == {"lidar", "labels", "trav_gt"}


def test_requires_two_datasets(ds_a):
    with pytest.raises(ValueError, match="at least two"):
        ZipDataset(ds_a)


def test_length_mismatch_raises():
    a = _DS(5, ["lidar"])
    b = _DS(3, ["trav_gt"])
    with pytest.raises(ValueError, match="same length"):
        ZipDataset(a, b)


def test_key_collision_raises(ds_a):
    b = _DS(5, ["lidar"])   # "lidar" already in ds_a
    with pytest.raises(KeyError, match="lidar"):
        ZipDataset(ds_a, b)


def test_key_collision_last_wins():
    a = _DS(3, ["lidar"], value_fn=lambda k, i: np.array([0.0]))
    b = _DS(3, ["lidar"], value_fn=lambda k, i: np.array([99.0]))
    z = ZipDataset(a, b, on_collision="last")
    np.testing.assert_array_equal(z[0].data["lidar"], [99.0])


def test_invalid_on_collision(ds_a, ds_b):
    with pytest.raises(ValueError, match="on_collision"):
        ZipDataset(ds_a, ds_b, on_collision="ignore")


# ------------------------------------------------------------------ access

def test_getitem_merges_data(ds_a, ds_b):
    z = ZipDataset(ds_a, ds_b)
    sample = z[0]
    assert "lidar" in sample.data
    assert "labels" in sample.data
    assert "trav_gt" in sample.data


def test_getitem_correct_values():
    a = _DS(3, ["x"], value_fn=lambda k, i: np.array([i * 10.0]))
    b = _DS(3, ["y"], value_fn=lambda k, i: np.array([i * 100.0]))
    z = ZipDataset(a, b)
    np.testing.assert_array_equal(z[2].data["x"],  [20.0])
    np.testing.assert_array_equal(z[2].data["y"], [200.0])


def test_three_datasets(ds_a, ds_b, ds_c):
    z = ZipDataset(ds_a, ds_b, ds_c)
    assert set(z[0].data.keys()) == {"lidar", "labels", "trav_gt", "ground_height"}


# ------------------------------------------------------------------ parent transforms

def test_parent_transforms_applied():
    a = _DS(3, ["lidar"], value_fn=lambda k, i: np.ones(3))
    b = _DS(3, ["trav_gt"])
    a.transform("lidar", lambda x: x * 5)
    z = ZipDataset(a, b)
    np.testing.assert_array_equal(z[0].data["lidar"], np.full(3, 5.0))


# ------------------------------------------------------------------ chaining

def test_join_sugar(ds_a, ds_b):
    z = ds_a.join(ds_b)
    assert isinstance(z, ZipDataset)
    assert len(z) == len(ds_a)


def test_join_then_transform(ds_a, ds_b):
    z = ds_a.join(ds_b).transform("lidar", lambda x: x * 0)
    np.testing.assert_array_equal(z[0].data["lidar"], np.zeros(3))


def test_join_then_filter(ds_a, ds_b):
    z = ds_a.join(ds_b).filter(lambda s: True)
    assert len(z) == len(ds_a)


def test_repr(ds_a, ds_b):
    z = ZipDataset(ds_a, ds_b)
    assert "ZipDataset" in repr(z)
    assert "5" in repr(z)
