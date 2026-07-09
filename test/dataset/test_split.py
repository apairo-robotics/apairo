import pytest

from apairo.dataset import split_sequences


def test_split_default_ratios():
    datasets = list(range(10))  # use ints as mock datasets
    train, val, test = split_sequences(datasets)
    assert len(train) == 8
    assert len(val) == 1
    assert len(test) == 1


def test_split_custom_ratios():
    datasets = list(range(10))
    train, val, test = split_sequences(datasets, ratios=(0.6, 0.2, 0.2))
    assert len(train) == 6
    assert len(val) == 2
    assert len(test) == 2


def test_split_no_overlap():
    datasets = list(range(10))
    train, val, test = split_sequences(datasets)
    all_items = train + val + test
    assert sorted(all_items) == sorted(datasets)


def test_split_bad_ratios():
    with pytest.raises(ValueError):
        split_sequences(list(range(5)), ratios=(0.5, 0.3, 0.3))
