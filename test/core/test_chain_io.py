"""A deployed chain saved as a JSON declaration and loaded back.

The file names classes and the values that rebuild them; loading imports
from trusted packages only and never runs code stored in the file. Anything
that cannot be declared is refused at save time, by name.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from apairo.core import Compose, Sample
from apairo.core.chain import load_chain
from apairo.core.chain_io import ChainDeclarationError
from apairo.core.interpolator import Interpolator
from apairo.dataset.stream import StreamDataset


class Lerp(Interpolator):
    def __call__(self, t, t0, v0, t1, v1):
        a = (t - t0) / (t1 - t0)
        return (1 - a) * v0 + a * v1


class Weights:
    """Rebuilt from its config, not its attributes (a model, say)."""

    input_keys = ["lidar"]
    output_keys = ["pred"]

    def __init__(self, path: str, scale: float = 1.0) -> None:
        self.path, self.scale = path, scale
        self.loaded = object()  # stands for weights: not declarable

    def get_config(self) -> dict:
        return {"path": self.path, "scale": self.scale}

    def __call__(self, s: Sample) -> Sample:
        s.data["pred"] = float(np.asarray(s.data["lidar"]).sum()) * self.scale
        return s


class Offset:
    def __init__(self, by: np.ndarray, pair: tuple) -> None:
        self.by, self.pair = by, pair

    def __call__(self, x):
        return x + self.by + self.pair[0]


class Stack:
    input_keys = ["lidar"]
    output_keys = ["lidar"]

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale

    def __call__(self, samples):
        data = dict(samples[-1].data)
        data["lidar"] = sum(np.asarray(s.data["lidar"]) for s in samples) * self.scale
        return Sample(data=data, timestamp=samples[-1].timestamp)


class Noisy:
    inference = False

    def __init__(self) -> None:
        self.rng = np.random.default_rng(0)

    def __call__(self, x):
        return x + self.rng.normal()


class HoldsAGenerator:
    def __init__(self) -> None:
        self.rng = np.random.default_rng(0)

    def __call__(self, x):
        return x


def double(x):
    return x * 2


def _stream(n: int = 6, offset: float = 0.0) -> StreamDataset:
    t = np.arange(n) * 0.1 + offset
    return StreamDataset(
        {
            "lidar": (t, [np.full(2, float(i)) for i in range(n)]),
            "pose": (t + 0.05, [np.full(2, 10.0 + i) for i in range(n)]),
        }
    )


@pytest.fixture(autouse=True)
def _no_in_place_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        yield


@pytest.fixture
def deployed():
    ds = _stream().synchronize(reference="lidar", method={"pose": Lerp()})
    ds = ds.window(size=2, reduce=Stack(scale=0.5))
    ds = ds.transform(
        "lidar",
        Compose([Offset(np.array([1.0, 2.0]), (3,)), double, Noisy()]),
        in_place=False,
    )
    ds = ds.transform(Weights("model.pt", scale=2.0), in_place=False)
    return ds.filter(np.array([0, 2])).inference(["pred", "pose"])


def _values(ds):
    return [(ds[i].data["pred"], ds[i].data["pose"].tolist()) for i in range(len(ds))]


def test_round_trip_gives_the_same_outputs(deployed, tmp_path):
    path = deployed.save(tmp_path / "chain.json")
    chain = load_chain(path, trusted=["test"])
    assert _values(chain.on(_stream(offset=7.0))) == _values(
        deployed.on(_stream(offset=7.0))
    )
    assert chain.outputs == ["pred", "pose"]
    assert chain.channels_needed() == ["lidar", "pose"]


def test_the_file_is_a_readable_declaration(deployed, tmp_path):
    doc = json.loads(deployed.save(tmp_path / "c.json").read_text())
    text = json.dumps(doc)
    assert doc["format"] == "apairo-chain"
    assert "test.core.test_chain_io:Weights" in text
    assert '"config"' in text and "model.pt" in text
    assert "Noisy" not in json.dumps(doc["nodes"])  # removed before saving
    assert "apairo" in doc["packages"]


def test_filters_are_not_saved(deployed, tmp_path):
    chain = load_chain(deployed.save(tmp_path / "c.json"), trusted=["test"])
    assert len(chain.on(_stream())) == len(deployed.on(_stream()))


def test_get_config_rebuilds_the_object(deployed, tmp_path):
    chain = load_chain(deployed.save(tmp_path / "c.json"), trusted=["test"])
    assert chain.on(_stream())[1].data["pred"] == deployed.on(_stream())[1].data["pred"]


def test_a_lambda_is_refused_by_name(tmp_path):
    ds = _stream().transform("lidar", lambda x: x, in_place=False)
    with pytest.raises(ChainDeclarationError, match="lambda.*module level"):
        ds.inference("lidar").save(tmp_path / "c.json")


def test_an_undeclarable_attribute_is_refused_by_name(tmp_path):
    ds = _stream().transform("lidar", HoldsAGenerator(), in_place=False)
    with pytest.raises(
        ChainDeclarationError, match=r"HoldsAGenerator\.rng.*get_config"
    ):
        ds.inference("lidar").save(tmp_path / "c.json")


def test_untrusted_classes_are_not_imported(deployed, tmp_path):
    path = deployed.save(tmp_path / "c.json")
    with pytest.raises(ChainDeclarationError, match=r"trusted=\['test'\]"):
        load_chain(path)


def test_not_a_chain_file(tmp_path):
    path = tmp_path / "c.json"
    path.write_text('{"format": "something"}')
    with pytest.raises(ChainDeclarationError, match="not an apairo chain"):
        load_chain(path)


def test_a_newer_format_is_refused(deployed, tmp_path):
    path = deployed.save(tmp_path / "c.json")
    doc = json.loads(path.read_text())
    doc["version"] = 99
    path.write_text(json.dumps(doc))
    with pytest.raises(ChainDeclarationError, match="newer apairo"):
        load_chain(path, trusted=["test"])


def test_other_package_versions_warn(deployed, tmp_path):
    path = deployed.save(tmp_path / "c.json")
    doc = json.loads(path.read_text())
    doc["packages"]["apairo"] = "0.0.1"
    path.write_text(json.dumps(doc))
    with pytest.warns(UserWarning, match="apairo 0.0.1"):
        load_chain(path, trusted=["test"])


def test_a_c_implemented_object_is_refused_not_emptied(tmp_path):
    ds = _stream().transform("lidar", np.sin, in_place=False)  # a C ufunc
    with pytest.raises(ChainDeclarationError, match="ufunc cannot be declared"):
        ds.inference("lidar").save(tmp_path / "c.json")


def test_a_partial_of_a_module_function_round_trips(tmp_path):
    import functools

    ds = _stream().transform("lidar", functools.partial(np.multiply, 3), in_place=False)
    with pytest.raises(ChainDeclarationError, match="ufunc cannot be declared"):
        ds.inference("lidar").save(tmp_path / "c.json")
    ds = _stream().transform("lidar", functools.partial(scaled, k=3.0), in_place=False)
    chain = load_chain(
        ds.inference("lidar").save(tmp_path / "c.json"), trusted=["test"]
    )
    np.testing.assert_array_equal(chain.on(_stream())[2].data["lidar"], np.full(2, 3.0))


def scaled(x, k=1.0):
    return x * k
