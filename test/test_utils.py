import numpy as np
import pytest

from apairo.utils.utils import dict_flatten, npy_analyser


@pytest.fixture
def nested_dict():
    return {"a": 1, "b": {"c": 2, "d": {"e": 3}}}


@pytest.fixture
def data_dict():
    return {
        "a": np.array([1, 2, 3]),
        "b": {"c": np.array([4, 5, 6]), "d": np.array([7, 8, 9])},
    }


def test_dict_flatten(nested_dict):
    expected = {"a": 1, "c": 2, "e": 3}
    assert dict_flatten(nested_dict) == expected

    expected = {"a": 1, "b.c": 2, "b.d.e": 3}

    def format_key(k, sk):
        return f"{k}.{sk}"

    assert dict_flatten(nested_dict, format_key) == expected, (
        "format_key argument is not working"
    )


def test_npy_analyser(tmp_path):
    # Mocking folder structure instead of relying on external path
    folder = tmp_path / "tartan2kitti"
    folder.mkdir()

    livox = folder / "livox"
    livox.mkdir()
    (livox / "000000.npy").touch()
    (livox / "000000_intensity.npy").touch()

    cmd = folder / "cmd"
    cmd.mkdir()
    (cmd / "000000.npy").touch()

    assert npy_analyser(str(livox)) == {"", "intensity"}
    assert npy_analyser(str(cmd)) == {""}
