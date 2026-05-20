import numpy as np
import pytest
from unittest.mock import MagicMock
from apairo.dataset.concat import ConcatDataset
from apairo.core.sample import Sample
import torch


def _make_mock_dataset(n: int, key: str = "imu"):
    ds = MagicMock()
    ds.keys = [key]
    ds.timestamps = {key: np.arange(n, dtype=float)}
    ds.__len__ = MagicMock(return_value=n)
    ds.__getitem__ = MagicMock(
        side_effect=lambda i: Sample(data={key: torch.zeros(3)}, timestamp=float(i))
    )
    ds.__iter__ = MagicMock(return_value=iter([
        Sample(data={key: torch.zeros(3)}, timestamp=float(i)) for i in range(n)
    ]))
    return ds


def test_len():
    a, b = _make_mock_dataset(5), _make_mock_dataset(3)
    cd = ConcatDataset([a, b])
    assert len(cd) == 8


def test_getitem_first_dataset():
    a, b = _make_mock_dataset(5), _make_mock_dataset(3)
    cd = ConcatDataset([a, b])
    cd[0]
    a.__getitem__.assert_called_with(0)


def test_getitem_second_dataset():
    a, b = _make_mock_dataset(5), _make_mock_dataset(3)
    cd = ConcatDataset([a, b])
    cd[5]
    b.__getitem__.assert_called_with(0)


def test_getitem_last_element():
    a, b = _make_mock_dataset(5), _make_mock_dataset(3)
    cd = ConcatDataset([a, b])
    cd[7]
    b.__getitem__.assert_called_with(2)


def test_getitem_out_of_range():
    cd = ConcatDataset([_make_mock_dataset(5)])
    with pytest.raises(IndexError):
        cd[5]


def test_iter_full_traversal():
    a, b = _make_mock_dataset(2), _make_mock_dataset(2)
    cd = ConcatDataset([a, b])
    items = list(cd)
    assert len(items) == 4


def test_unified_timestamps():
    a = _make_mock_dataset(3, "imu")
    b = _make_mock_dataset(2, "imu")
    cd = ConcatDataset([a, b])
    assert "imu" in cd.timestamps
    assert len(cd.timestamps["imu"]) == 5


def test_timestamps_none_for_sync_datasets():
    ds = MagicMock()
    ds.keys = ["lidar"]
    ds.timestamps = None
    ds.__len__ = MagicMock(return_value=3)
    cd = ConcatDataset([ds, ds])
    assert cd.timestamps is None
    assert cd.is_synchronous is True
