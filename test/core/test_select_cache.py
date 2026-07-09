import numpy as np
import pytest

from apairo.core import AbstractDataset, CachedDataset, ChannelView
from apairo.core.sample import Sample


class _DS(AbstractDataset):
    def __init__(self, n=4):
        self._keys = ["lidar", "trav_gt", "ground_height"]
        self._n = n

    def __len__(self):
        return self._n

    def _load(self, idx):
        return Sample(
            data={
                "lidar": np.full((3, 3), float(idx)),
                "trav_gt": np.array([idx % 2]),
                "ground_height": np.array([idx * 10.0]),
            }
        )


@pytest.fixture
def ds():
    return _DS()


# ------------------------------------------------------------------ select


def test_select_restricts_keys(ds):
    view = ds.select(["lidar", "trav_gt"])
    assert isinstance(view, ChannelView)
    sample = view[0]
    assert set(sample.data.keys()) == {"lidar", "trav_gt"}
    assert "ground_height" not in sample.data


def test_select_len(ds):
    assert len(ds.select(["lidar"])) == len(ds)


def test_select_inherits_parent_transforms(ds):
    ds.transform("ground_height", lambda x: x * 99)
    view = ds.select(["ground_height"])
    np.testing.assert_array_equal(view[0].data["ground_height"], [0.0 * 99])
    np.testing.assert_array_equal(view[1].data["ground_height"], [10.0 * 99])


def test_select_missing_key_raises(ds):
    view = ds.select(["nonexistent"])
    with pytest.raises(KeyError):
        view[0]


def test_select_chaining(ds):
    view = ds.select(["lidar"]).transform("lidar", lambda x: x * 0)
    np.testing.assert_array_equal(view[0].data["lidar"], np.zeros((3, 3)))


def test_select_repr(ds):
    assert "ChannelView" in repr(ds.select(["lidar"]))


# ------------------------------------------------------------------ cache


def test_cache_len(ds):
    cached = ds.cache()
    assert isinstance(cached, CachedDataset)
    assert len(cached) == len(ds)


def test_cache_data_correct(ds):
    cached = ds.cache()
    for i in range(len(ds)):
        np.testing.assert_array_equal(cached[i].data["lidar"], ds[i].data["lidar"])


def test_cache_no_mutation(ds):
    cached = ds.cache()
    s1 = cached[0]
    s1.data["lidar"][:] = 999  # mutate the returned sample
    s2 = cached[0]  # fetch again
    assert s2.data["lidar"][0, 0] != 999  # cache is not corrupted


def test_cache_after_select(ds):
    ds.transform("ground_height", lambda x: x * 2)
    cached = ds.select(["ground_height"]).cache()
    assert set(cached[0].data.keys()) == {"ground_height"}
    np.testing.assert_array_equal(cached[1].data["ground_height"], [20.0])


def test_cache_then_join(ds):
    ds_prior = ds.select(["ground_height"]).cache()
    ds_base = _DS()
    ds_base._keys = ["lidar", "trav_gt"]
    combined = ds_base.join(ds_prior)
    sample = combined[0]
    assert "lidar" in sample.data
    assert "ground_height" in sample.data


def test_cache_supports_transform(ds):
    cached = ds.select(["lidar"]).cache()
    cached.transform("lidar", lambda x: x * 5)
    np.testing.assert_array_equal(cached[1].data["lidar"], np.full((3, 3), 5.0))


def test_cache_repr(ds):
    assert "CachedDataset" in repr(ds.cache())
