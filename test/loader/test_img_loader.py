import numpy as np
import pytest
from PIL import Image

from apairo.loader import IMGLoader
from test.utils import create_random_images


@pytest.fixture
def img_loader_data(tmp_path):
    directory = tmp_path / "img_loader_test"
    directory.mkdir()
    create_random_images(100, (16, 16), str(directory))
    return directory


def test_len(img_loader_data):
    loader = IMGLoader(str(img_loader_data))
    assert loader.shape == (16, 16, 3)  # HWC -- PIL convention


def test_iteration(img_loader_data):
    directory = img_loader_data
    loader = IMGLoader(str(directory))
    image0_path = directory / "000000.png"
    expected = np.array(Image.open(str(image0_path)))
    assert np.allclose(loader[0], expected)
