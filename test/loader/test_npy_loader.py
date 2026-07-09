import numpy as np
import pytest

from apairo.loader.npy_loader import NPYLoader
from test.utils import create_npy_file


@pytest.fixture
def npy_loader_data(tmp_path):
    data = np.array([1, 2, 3, 4, 5])
    directory = tmp_path / "npy_loader_test"
    directory.mkdir()
    create_npy_file(data, filename="data.npy", directory=str(directory))
    return data, directory


def test_len(npy_loader_data):
    data, directory = npy_loader_data
    loader = NPYLoader(str(directory))
    assert loader.array.shape == data.shape


def test_getitem(npy_loader_data):
    data, directory = npy_loader_data
    loader = NPYLoader(str(directory))
    # Default format is ""
    assert np.allclose(loader[0], data[0])


def test_shape(npy_loader_data):
    data, directory = npy_loader_data
    loader = NPYLoader(str(directory))
    assert loader.shape == (1,)  # Tuple comparison
