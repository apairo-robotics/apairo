"""transform()'s in_place default is deprecated: it changes in 0.11.

Left unset, in_place behaves as True (the 0.9 behaviour) and warns. Passing
it either way is silent. Assigning the result gives the same dataset under
both values, which is why the docs write ``ds = ds.transform(...)``.
"""

import warnings

import numpy as np
import pytest

from apairo.core import AbstractDataset, Sample

LIDAR = np.array([[1.0, 2.0], [3.0, 4.0]])


class TinyDataset(AbstractDataset):
    def __init__(self, n: int = 3):
        self.keys = ["lidar"]
        self.loaders = {}
        self._n = n

    def __len__(self):
        return self._n

    def _load(self, idx):
        return Sample(data={"lidar": LIDAR.copy()})


def _plus_one(x):
    return x + 1


def _times_ten(x):
    return x * 10


@pytest.fixture
def ds():
    return TinyDataset()


@pytest.fixture
def no_future_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        yield


def lidar(dataset, idx=0):
    return dataset[idx].data["lidar"]


# ---------------------------------------------------------------- unset


def test_unset_warns_and_names_the_fix(ds):
    with pytest.warns(FutureWarning, match=r"ds = ds\.transform") as record:
        ds.transform("lidar", _plus_one)
    assert "0.11" in str(record[0].message)


def test_unset_warning_points_at_the_caller(ds):
    with pytest.warns(FutureWarning) as record:
        ds.transform("lidar", _plus_one)
    assert record[0].filename == __file__


def test_unset_keeps_the_09_behaviour(ds):
    with pytest.warns(FutureWarning):
        out = ds.transform("lidar", _plus_one)
    assert out is ds
    np.testing.assert_array_equal(lidar(ds), LIDAR + 1)


def test_unset_warns_for_every_form(ds):
    with pytest.warns(FutureWarning):
        ds.transform(lambda s: s)
    with pytest.warns(FutureWarning):
        ds.transform("lidar", _plus_one, output="lidar_f", keep=False)


# ------------------------------------------------------------- explicit


def test_in_place_true_mutates_silently(ds, no_future_warning):
    out = ds.transform("lidar", _plus_one, in_place=True)
    assert out is ds
    np.testing.assert_array_equal(lidar(ds), LIDAR + 1)


def test_in_place_false_returns_a_copy_silently(ds, no_future_warning):
    out = ds.transform("lidar", _plus_one, in_place=False)
    assert out is not ds
    np.testing.assert_array_equal(lidar(out), LIDAR + 1)
    np.testing.assert_array_equal(lidar(ds), LIDAR)


def test_in_place_false_variants_are_independent(ds, no_future_warning):
    a = ds.transform("lidar", _plus_one, in_place=False)
    b = ds.transform("lidar", _times_ten, in_place=False)
    np.testing.assert_array_equal(lidar(a), LIDAR + 1)
    np.testing.assert_array_equal(lidar(b), LIDAR * 10)


def test_in_place_false_copy_does_not_see_later_steps(ds, no_future_warning):
    out = ds.transform("lidar", _plus_one, in_place=False)
    ds.transform("lidar", _times_ten, in_place=True)
    np.testing.assert_array_equal(lidar(out), LIDAR + 1)


# ----------------------------------------------------------- assigning


@pytest.mark.parametrize("in_place", [True, False])
def test_assigned_chain_gives_the_same_result_either_way(ds, in_place):
    out = ds.transform("lidar", _plus_one, in_place=in_place)
    out = out.transform("lidar", _times_ten, in_place=in_place)
    np.testing.assert_array_equal(lidar(out), (LIDAR + 1) * 10)


def test_in_place_false_keeps_drop_keys_on_the_copy(ds, no_future_warning):
    out = ds.transform("lidar", _plus_one, output="_tmp", keep=False, in_place=False)
    assert "_tmp" not in out[0].data
    assert not getattr(ds, "_drop_keys", set())
