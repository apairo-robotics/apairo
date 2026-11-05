"""inference(): the chain of a dataset as deployed, and .on(): on another source.

The chain is read back from how the dataset was built. inference(outputs)
keeps what the outputs need, removes training-only steps, keeps frame filters
on the original data and drops caches and repeats. on(source) calls every
operation again on a new source -- the same frames, read from disk or from a
stream, must give the same outputs.
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pytest

from apairo.core import AbstractDataset, Compose, Sample
from apairo.core.chain import InferenceView, step_io
from apairo.core.filtered_view import FilteredView
from apairo.core.interpolator import Interpolator
from apairo.dataset.raw import RawDataset
from apairo.dataset.stream import StreamDataset

# ------------------------------------------------------------------ steps


class Scale:
    def __init__(self, k: float) -> None:
        self.k = k

    def __call__(self, x):
        return x * self.k


class Shift:
    """An augmentation: removed by inference()."""

    inference = False

    def __call__(self, x):
        return x + 1000.0


class Model:
    """A sample-level step that declares what it reads and writes."""

    input_keys = ["lidar"]
    output_keys = ["pred"]

    def __call__(self, s: Sample) -> Sample:
        s.data["pred"] = float(np.asarray(s.data["lidar"]).sum())
        return s


class Stack:
    """A window reducer: sums the window's lidar into the anchor's."""

    input_keys = ["lidar"]
    output_keys = ["lidar"]

    def __call__(self, samples):
        anchor = samples[-1]
        data = dict(anchor.data)
        data["lidar"] = sum(np.asarray(s.data["lidar"]) for s in samples)
        return Sample(data=data, timestamp=anchor.timestamp)


class Undeclared:
    def __call__(self, samples):
        return samples[-1]


class Lerp(Interpolator):
    def __call__(self, t, t0, v0, t1, v1):
        a = (t - t0) / (t1 - t0)
        return (1 - a) * v0 + a * v1


def opaque(s: Sample) -> Sample:
    return s


def label(x):
    return x * 2


# ------------------------------------------------------------------- data

LIDAR_T = np.arange(6) * 0.1
POSE_T = LIDAR_T + 0.01


def _lidar(i: int) -> np.ndarray:
    return np.full(2, float(i))


def _stream(n: int = 6, offset: float = 0.0) -> StreamDataset:
    t = np.arange(n) * 0.1 + offset
    return StreamDataset(
        {
            "lidar": (t, [_lidar(i) for i in range(n)]),
            "pose": (t + 0.01, [np.full(2, 10.0 + i) for i in range(n)]),
            "gt": (t, [np.full(2, -1.0) for _ in range(n)]),
        }
    )


def _channel(seq, name, stamps, values):
    d = seq / name
    d.mkdir(parents=True)
    for i, v in enumerate(values):
        np.save(d / f"{i:06d}.npy", v)
    np.savetxt(d / "timestamps.txt", stamps, fmt="%.6f")


@pytest.fixture
def raw(tmp_path):
    """Two sequences holding the same frames as _stream()."""
    for name in ("s1", "s2"):
        seq = tmp_path / name
        _channel(seq, "lidar", 100 + LIDAR_T, [_lidar(i) for i in range(6)])
        _channel(seq, "pose", 100 + POSE_T, [np.full(2, 10.0 + i) for i in range(6)])
        _channel(seq, "gt", 100 + LIDAR_T, [np.full(2, -1.0) for _ in range(6)])
    return RawDataset(tmp_path, keys=["lidar", "pose", "gt"])


@pytest.fixture
def train():
    ds = _stream().synchronize(reference="lidar")
    ds = ds.transform("lidar", Compose([Scale(2.0), Shift()]), in_place=False)
    ds = ds.transform("gt", label, output="gt2", in_place=False)
    return ds.transform(Model(), in_place=False)


@pytest.fixture(autouse=True)
def _no_in_place_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        yield


def _values(ds, key="pred"):
    return [ds[i].data[key] for i in range(len(ds))]


# ---------------------------------------------------------------- pruning


def test_removes_augmentations_and_keeps_the_rest(train):
    view = train.inference(["pred"])
    assert isinstance(view, InferenceView)
    # Scale(2) kept, Shift removed: pred = 2 * (i + i)
    assert _values(view) == [4.0 * i for i in range(1, 6)]
    assert _values(train)[0] != _values(view)[0]


def test_samples_hold_the_outputs_only(train):
    view = train.inference(["pred", "gt"])
    assert set(view[0].data) == {"pred", "gt"}
    assert view.keys == ["pred", "gt"]


def test_summary_names_what_was_kept_and_removed(train):
    summary = train.inference("pred").summary
    removed = {(e.item, e.reason) for e in summary.removed}
    assert ("Shift", "training only") in removed
    assert ("label('gt' -> 'gt2')", "does not lead to the outputs") in removed
    assert {e.item for e in summary.kept} == {"Compose([Scale])('lidar')", "Model"}
    assert "removed" in str(summary) and "Shift" in str(summary)


def test_a_step_feeding_a_kept_step_is_kept():
    ds = _stream().synchronize(reference="lidar")
    ds = ds.transform("lidar", Scale(3.0), output="mid", in_place=False)
    ds = ds.transform("mid", Scale(2.0), output="out", in_place=False)
    ds = ds.transform("gt", label, output="unused", in_place=False)
    view = ds.inference("out")
    assert [e.item for e in view.summary.kept] == [
        "Scale('lidar' -> 'mid')",
        "Scale('mid' -> 'out')",
    ]
    np.testing.assert_array_equal(view[0].data["out"], _lidar(1) * 6)


def test_an_opaque_step_keeps_everything_before_it():
    ds = _stream().synchronize(reference="lidar")
    ds = ds.transform("gt", label, output="unused", in_place=False)
    ds = ds.transform(opaque, in_place=False)
    ds = ds.transform(Model(), in_place=False)
    summary = ds.inference("pred").summary
    assert "label('gt' -> 'unused')" in {e.item for e in summary.kept}
    assert any(e.item == "opaque" and "opaque" in e.reason for e in summary.flags)


def test_unknown_output_is_refused(train):
    with pytest.raises(ValueError, match=r"\['nope'\].*Available"):
        train.inference(["nope"])


def test_compose_of_augmentations_only_is_removed_whole():
    ds = _stream().synchronize(reference="lidar")
    ds = ds.transform("lidar", Compose([Shift(), Shift()]), in_place=False)
    view = ds.inference("lidar")
    assert [e.item for e in view.summary.removed] == ["Shift", "Shift"]
    np.testing.assert_array_equal(view[0].data["lidar"], _lidar(1))


def test_step_io_reads_declarations():
    ds = _stream().transform("lidar", label, output="x", in_place=False)
    assert step_io(ds._pipeline[-1]) == ({"lidar"}, {"x"})
    assert step_io(Model()) == ({"lidar"}, {"pred"})
    assert step_io(opaque) == (None, None)
    assert step_io(Compose([Model(), Model()])) == ({"lidar"}, {"pred"})


# ---------------------------------------------------------------- snapshot


def test_is_a_snapshot(train):
    view = train.inference("pred")
    before = _values(view)
    train.transform("pred", lambda p: p + 1, in_place=True)
    assert _values(view) == before


def test_leaves_the_training_chain_untouched(train):
    before = _values(train)
    steps = list(train._pipeline)
    train.inference("pred")
    assert train._pipeline == steps
    assert _values(train) == before


def test_an_inference_view_pickles(train):
    view = train.inference("pred")
    clone = pickle.loads(pickle.dumps(view))
    assert _values(clone) == _values(view)


# ------------------------------------------------- filters, cache, repeat


def test_filters_stay_on_the_original_data(train):
    # frame k of train is lidar frame k + 1 (lidar 0 has no earlier pose)
    view = train.filter(np.array([1, 3])).inference("pred")
    assert _values(view) == [8.0, 16.0]
    assert any(e.where == "filter" for e in view.summary.flags)


def test_cache_and_repeat_are_removed(train):
    view = train.cache().repeat(3).inference("pred")
    assert len(view) == len(train)
    assert {e.item for e in view.summary.removed} >= {"cache", "repeat"}


def test_augmentation_before_a_cache_is_seen():
    ds = _stream().synchronize(reference="lidar")
    ds = ds.transform("lidar", Shift(), in_place=False).cache()
    view = ds.inference("lidar")
    assert "Shift('lidar')" in {e.item for e in view.summary.removed}
    np.testing.assert_array_equal(view[0].data["lidar"], _lidar(1))


# ------------------------------------------------------------------ window


def test_window_with_a_declared_reducer():
    ds = _stream().synchronize(reference="lidar").window(size=2, reduce=Stack())
    ds = ds.transform("gt", label, output="unused", in_place=False)
    view = ds.inference("lidar")
    np.testing.assert_array_equal(view[1].data["lidar"], _lidar(1) + _lidar(2))
    assert not view.summary.flags


def test_window_with_an_undeclared_reducer_is_flagged():
    ds = _stream().synchronize(reference="lidar").window(size=2, reduce=Undeclared())
    flags = ds.inference("lidar").summary.flags
    assert any(e.where == "window" for e in flags)


# --------------------------------------------------------------- causality


def test_non_causal_synchronize_is_flagged():
    ds = _stream().synchronize(reference="lidar", method={"pose": Lerp()})
    view = ds.inference("pose")
    assert [e.reason for e in view.summary.flags] == ["non-causal"]


def test_causal_refuses_a_non_causal_chain():
    ds = _stream().synchronize(reference="lidar", method="nearest")
    with pytest.raises(ValueError, match="causal=True"):
        ds.inference("lidar", causal=True)
    _stream().synchronize(reference="lidar").inference("lidar", causal=True)


def test_reference_none_is_resolved_and_flagged():
    view = _stream().synchronize().inference("lidar")
    assert any("reference=" in e.item for e in view.summary.flags)


# ---------------------------------------------------------------------- on


def test_on_gives_the_same_outputs_from_disk_and_from_a_stream(raw):
    """The parity test: the chain built on files, replayed on a stream of
    the same frames."""
    ds = raw.synchronize(reference="lidar", method={"pose": "previous"})
    ds = ds.window(size=2, reduce=Stack())
    ds = ds.transform("lidar", Compose([Scale(2.0), Shift()]), in_place=False)
    ds = ds.transform(Model(), in_place=False)
    deployed = ds.inference("pred")
    live = deployed.on(_stream())
    per_sequence = len(deployed) // 2
    assert len(live) == per_sequence
    assert _values(live) == _values(deployed)[:per_sequence]


def test_on_drops_filters(train):
    view = train.filter(np.array([1])).inference("pred")
    assert len(view) == 1
    assert len(view.on(_stream())) == len(train)


class Frames(AbstractDataset):
    """A synchronous in-memory source: frame i holds lidar i."""

    def __init__(self, n: int = 4) -> None:
        self.keys = ["lidar", "gt"]
        self._n = n

    def __len__(self):
        return self._n

    def _load(self, idx):
        return Sample(data={"lidar": _lidar(idx), "gt": np.zeros(2)})


def test_on_carries_the_source_steps_over():
    src = Frames().transform("lidar", Scale(10.0), in_place=False)
    view = src.select(["lidar"]).inference("lidar")
    np.testing.assert_array_equal(view.on(Frames())[2].data["lidar"], _lidar(2) * 10)
    # the new source's own steps run first, then the chain's
    halved = Frames().transform("lidar", Scale(0.5), in_place=False)
    np.testing.assert_array_equal(view.on(halved)[2].data["lidar"], _lidar(2) * 5)


def test_synchronize_ignores_the_steps_of_its_source():
    src = _stream().transform("lidar", Scale(10.0), in_place=False)
    ds = src.synchronize(reference="lidar")
    # synchronize reads raw channels: steps on its parent never apply
    view = ds.transform(Model(), in_place=False).inference("pred")
    other = _stream().transform("lidar", Scale(0.5), in_place=False)
    np.testing.assert_allclose(_values(view.on(other)), _values(view))


def test_on_keeps_steps_registered_after_synchronize():
    ds = _stream().synchronize(reference="lidar")
    base = ds.inference(["lidar", "pose"])
    synced = _stream().synchronize(reference="lidar")
    chain = synced.transform("lidar", Scale(3.0), in_place=False).inference("lidar")
    np.testing.assert_array_equal(
        chain.on(_stream())[0].data["lidar"], base[0].data["lidar"] * 3
    )


def test_on_needs_the_channels_the_chain_reads(train):
    deployed = train.inference("pred")
    assert deployed.chain.channels_needed() == ["lidar"]
    lidar_only = StreamDataset({"lidar": (LIDAR_T, [_lidar(i) for i in range(6)])})
    assert len(deployed.on(lidar_only)) == 6
    gt_only = StreamDataset({"gt": (LIDAR_T, [_lidar(i) for i in range(6)])})
    with pytest.raises(ValueError, match=r"lacks channels.*\['lidar'\]"):
        deployed.on(gt_only)


def test_a_second_source_is_flagged_and_on_refuses(train):
    prior = _stream().synchronize(reference="lidar").select(["pose"])
    prior = prior.transform("pose", label, output="prior", in_place=False)
    joined = train.join(prior.select(["prior"]))
    view = joined.inference(["pred", "prior"])
    assert any(e.where == "join" for e in view.summary.flags)
    with pytest.raises(ValueError, match="2"):
        view.on(_stream())


def test_a_view_built_directly_is_opaque(train):
    view = FilteredView(train, np.array([0, 1])).inference("pred")
    assert any("built directly" in e.reason for e in view.summary.flags)
    with pytest.raises(ValueError, match="built directly"):
        view.on(_stream())


def test_inference_of_an_inference_view(train):
    view = train.inference(["pred", "gt"]).inference("pred")
    assert _values(view) == _values(train.inference("pred"))


# -------------------------------------------------------------------- save


def test_save_and_load_give_the_same_outputs(train, tmp_path):
    from apairo.core.chain import load_chain

    deployed = train.inference("pred")
    path = deployed.save(tmp_path / "chain.json")
    chain = load_chain(path, trusted=["test"])
    assert _values(chain.on(_stream())) == _values(deployed.on(_stream()))
    assert [e.item for e in chain.summary.removed] == [
        e.item for e in deployed.summary.removed
    ]


def test_a_pickled_cache_does_not_carry_its_parent(train):
    cached = train.cache()
    assert cached._op is not None
    clone = pickle.loads(pickle.dumps(cached))
    assert clone._op is None
    assert _values(clone) == _values(cached)


def test_chain_reference_lookback_and_causality():
    ds = (
        _stream()
        .synchronize(reference="lidar")
        .window(size=3, stride=2, reduce=Stack())
    )
    chain = ds.inference("lidar").chain
    assert (chain.reference, chain.lookback, chain.causal) == ("lidar", 4, True)
    lerp = _stream().synchronize(reference="pose", method={"lidar": Lerp()})
    assert lerp.inference("lidar").chain.causal is False
    assert Frames().select(["lidar"]).inference("lidar").chain.reference is None
