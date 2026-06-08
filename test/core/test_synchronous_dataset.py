import numpy as np
import pytest
from pathlib import Path
from apairo.core.synchronous_dataset import SynchronousDataset
from apairo.core.sample import Sample


class MockSyncDataset(SynchronousDataset):
    def __init__(self, n: int):
        self._keys = ["lidar", "label"]
        self._n = n

    def __len__(self) -> int:
        return self._n

    def _load(self, idx: int) -> Sample:
        if not 0 <= idx < self._n:
            raise IndexError(idx)
        return Sample(data={"lidar": np.zeros((100, 4)), "label": np.zeros(100)})

class MockSyncDatasetWithFiles(SynchronousDataset):
    def __init__(self, root: Path, files: dict[str, list[Path]]):
        self._root = root
        self._files = files
        self._n = len(next(iter(files.values())))

    def __len__(self) -> int:
        return self._n

    def _load(self, idx: int) -> Sample:
        return Sample(data={})

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


def test_root_dir(tmp_path):
    files = {"lidar": [tmp_path / "seq0" / "lidar" / "000000.bin"]}
    ds = MockSyncDatasetWithFiles(tmp_path, files)
    assert ds.root_dir == tmp_path
    assert isinstance(ds.root_dir, Path)


def test_derived_path_structure(tmp_path):
    lidar_path = tmp_path / "seq0" / "lidar" / "000000.bin"
    files = {"lidar": [lidar_path]}
    ds = MockSyncDatasetWithFiles(tmp_path, files)
    result = ds.derived_path(0, "elevation_map", "npy")
    assert result.stem == "000000"
    assert result.suffix == ".npy"
    assert result.parent.name == "elevation_map"
