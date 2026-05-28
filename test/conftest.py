import pytest
import numpy as np
import matplotlib.pyplot as plt

from apairo.loader import (
    NPYLoader,
    NPYSLoader,
    IMGLoader,
)


@pytest.fixture
def npy_file(tmp_path_factory):
    def _npy_file(data, filename, directory):
        path = tmp_path_factory.mktemp(directory) / filename
        np.save(path, data)
        return path

    return _npy_file


@pytest.fixture
def npy_loader(npy_file):
    def _npy_loader(data, filename, directory):
        path = npy_file(data, filename, directory)
        loader = NPYLoader(path.parent)
        return loader

    return _npy_loader


@pytest.fixture
def npys_loader(npy_file):
    def _npys_loader(shape, directory, file_spec=""):
        dim = shape[0]
        for i in range(dim):
            data = np.random.rand(*shape[1:])
            if file_spec:
                path = npy_file(data, f"{i:06}_{file_spec}.npy", directory)
            else:
                path = npy_file(data, f"{i:06}.npy", directory)
        loader = NPYSLoader(path.parent)
        return loader

    return _npys_loader


@pytest.fixture
def img_loader(tmp_path_factory):
    def _img_loader(shape, directory):
        dim = shape[0]
        for i in range(dim):
            data = np.random.rand(*shape[1:])
            path = tmp_path_factory.mktemp(directory) / f"{i:06}.png"
            plt.imsave(path, data)
        loader = IMGLoader(path.parent)
        return loader

    return _img_loader
