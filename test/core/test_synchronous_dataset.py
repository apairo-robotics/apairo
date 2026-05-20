import torch
import pytest
from apairo.core.synchronous_dataset import SynchronousDataset
from apairo.core.sample import Sample


class MockSyncDataset(SynchronousDataset):
    def __init__(self, n: int):
        self._keys = ["lidar", "label"]
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> Sample:
        if not 0 <= idx < self._n:
            raise IndexError(idx)
        return Sample(data={"lidar": torch.zeros(100, 4), "label": torch.zeros(100)})

    def __iter__(self):
        self._pos = 0
        return self

    def __next__(self) -> Sample:
        if self._pos >= self._n:
            raise StopIteration
        s = self[self._pos]
        self._pos += 1
        return s


def test_no_timestamps():
    ds = MockSyncDataset(5)
    assert ds.timestamps is None
    assert ds.is_synchronous is True


def test_len():
    assert len(MockSyncDataset(5)) == 5


def test_getitem_returns_sample():
    s = MockSyncDataset(5)[0]
    assert isinstance(s, Sample)
    assert s.timestamp is None
    assert "lidar" in s.data


def test_iter():
    assert len(list(MockSyncDataset(3))) == 3


def test_out_of_range():
    with pytest.raises(IndexError):
        MockSyncDataset(3)[3]
