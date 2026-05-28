import numpy as np
import pytest

from apairo.writer import NPYWriter, BINWriter, WRITERS


@pytest.fixture
def arr():
    return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)


class TestNPYWriter:
    def test_creates_file(self, tmp_path, arr):
        path = tmp_path / "out.npy"
        NPYWriter().write(arr, path)
        assert path.exists()

    def test_creates_parent_dirs(self, tmp_path, arr):
        path = tmp_path / "a" / "b" / "c" / "out.npy"
        NPYWriter().write(arr, path)
        assert path.exists()

    def test_content(self, tmp_path, arr):
        path = tmp_path / "out.npy"
        NPYWriter().write(arr, path)
        loaded = np.load(path)
        np.testing.assert_array_almost_equal(loaded, arr)


class TestBINWriter:
    def test_creates_file(self, tmp_path, arr):
        path = tmp_path / "out.bin"
        BINWriter().write(arr, path)
        assert path.exists()

    def test_creates_parent_dirs(self, tmp_path, arr):
        path = tmp_path / "p" / "q" / "out.bin"
        BINWriter().write(arr, path)
        assert path.exists()

    def test_content(self, tmp_path, arr):
        path = tmp_path / "out.bin"
        BINWriter().write(arr, path)
        loaded = np.fromfile(path, dtype=np.float32).reshape(arr.shape)
        np.testing.assert_array_almost_equal(loaded, arr)


class TestWRITERS:
    def test_npy_key(self):
        assert WRITERS["npy"] is NPYWriter

    def test_npys_key(self):
        assert WRITERS["npys"] is NPYWriter

    def test_bin_key(self):
        assert WRITERS["bin"] is BINWriter

    def test_unknown_key_raises(self):
        with pytest.raises(KeyError):
            _ = WRITERS["xyz"]
