import numpy as np
import pytest

from apairo.core.utils.exceptions import FileExtensionError
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


# ─────────────────── explicit file selection (colocated arrays) ───────────────


@pytest.fixture
def colocated_arrays(tmp_path):
    """A directory holding two stacked arrays, as gicp_poses/ colocates poses.npy
    and valid_mask.npy."""
    directory = tmp_path / "gicp_poses"
    directory.mkdir()
    poses = np.arange(12).reshape(3, 4)
    mask = np.array([True, False, True])
    create_npy_file(poses, filename="poses.npy", directory=str(directory))
    create_npy_file(mask, filename="valid_mask.npy", directory=str(directory))
    return directory, poses, mask


def test_file_selects_the_named_array(colocated_arrays):
    directory, poses, mask = colocated_arrays
    # Without `file`, glob picks poses.npy ('.' sorts before '_') -- the second
    # array is unreachable; with `file` each is addressable by name.
    np.testing.assert_array_equal(NPYLoader(directory).array, poses)
    np.testing.assert_array_equal(
        NPYLoader(directory, file="valid_mask.npy").array, mask
    )
    np.testing.assert_array_equal(NPYLoader(directory, file="poses.npy").array, poses)


def test_file_missing_raises(colocated_arrays):
    directory, _, _ = colocated_arrays
    with pytest.raises(FileExtensionError, match="No such .npy file"):
        NPYLoader(directory, file="absent.npy")
