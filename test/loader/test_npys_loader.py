import numpy as np
import pytest

from apairo.loader.npys_loader import NPYSLoader
from apairo.utils import npy_analyser
from test.utils import create_random_npy_files


@pytest.fixture
def npys_loader_data(tmp_path):
    directory = tmp_path / "npys_loader_test"

    # Needs to be created
    directory.mkdir()
    create_random_npy_files(100, (5, 5), str(directory))
    create_random_npy_files(100, (5, 5), str(directory), "intensity")
    return directory


def test_len(npys_loader_data):
    loader = NPYSLoader(str(npys_loader_data))
    assert len(loader) == 100


def test_default_ignores_suffixed_variants(npys_loader_data):
    loader = NPYSLoader(str(npys_loader_data))
    assert all("_" not in f for f in loader.files)


def test_getitem(npys_loader_data):
    loader = NPYSLoader(str(npys_loader_data))
    file0 = np.load(str(npys_loader_data / "000000.npy"))
    assert np.allclose(loader[0], file0)


def test_dataset_imposed_files(npys_loader_data):
    """The dataset resolves the per-channel file list and injects it."""
    import os

    intensity_files = sorted(
        f for f in os.listdir(npys_loader_data) if f.endswith("_intensity.npy")
    )
    loader = NPYSLoader(str(npys_loader_data), files=intensity_files)
    assert len(loader) == 100

    file0 = np.load(str(npys_loader_data / "000000_intensity.npy"))
    assert np.allclose(loader[0], file0)


def test_npy_analyser_discovers_formats(npys_loader_data):
    assert npy_analyser(str(npys_loader_data)) == {"", "intensity"}


def test_empty_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        NPYSLoader(str(tmp_path))
