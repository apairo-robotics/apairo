"""A preprocessor with several inputs on an asynchronous dataset.

An asynchronous dataset iterates its interleaved event timeline, one channel per
sample, so a multi-input preprocessor used to die on the first frame with a
``KeyError``: the sample never held both inputs. When the inputs share one clock
-- a channel derived through ``timestamps_from``, or identical timestamps -- the
runner now zips them row for row. Inputs on different clocks are refused by
name: pairing them is a synchronisation, not something the runner can guess.
"""

from collections.abc import Iterator

import numpy as np
import pytest

from apairo.core.config import read_config, write_config
from apairo.core.preprocessor import FramePreprocessor, SequencePreprocessor
from apairo.core.sample import Sample
from apairo.dataset.raw import RawDataset

N = 4
CLOCK = 100.0 + np.arange(N) * 0.1


class _Sum(FramePreprocessor):
    output_key = "sum"
    output_loader = "npys"
    input_keys = ["lidar", "trav"]
    timestamps_from = "lidar"
    sources = ["lidar", "trav"]

    def __call__(self, sample: Sample) -> np.ndarray:
        return sample.data["lidar"] + sample.data["trav"]


class _SumAndDiff(FramePreprocessor):
    output_keys = ["plus", "minus"]
    output_loader = "npys"
    input_keys = ["lidar", "trav"]

    def __call__(self, sample: Sample) -> dict:
        a, b = sample.data["lidar"], sample.data["trav"]
        return {"plus": a + b, "minus": b - a}


class _SumThree(FramePreprocessor):
    output_key = "sum3"
    output_loader = "npys"
    input_keys = ["lidar", "trav", "cmd"]

    def __call__(self, sample: Sample) -> np.ndarray:
        return sample.data["lidar"] + sample.data["trav"]


class _StackedSum(SequencePreprocessor):
    output_key = "stacked"
    output_loader = "npy"
    input_keys = ["lidar", "trav"]

    def __call__(self, frames: Iterator[Sample]) -> np.ndarray:
        return np.stack([s.data["lidar"] + s.data["trav"] for s in frames])


class _PerFrameSum(SequencePreprocessor):
    output_key = "rowsum"
    output_loader = "npys"
    input_keys = ["lidar", "trav"]

    def __call__(self, frames: Iterator[Sample]) -> np.ndarray:
        return np.stack([s.data["lidar"] + s.data["trav"] for s in frames])


def _frames(seq, name, n, base, timestamps=None):
    d = seq / name
    d.mkdir(parents=True)
    for i in range(n):
        np.save(d / f"{i:06d}.npy", np.full(3, base + i, np.float32))
    if timestamps is not None:
        np.savetxt(d / "timestamps.txt", timestamps)


def _sequence(seq, *, trav_clock="borrowed", cmd=False):
    """``lidar`` (row i holds i) and ``trav`` (row i holds 10 + i).

    *trav_clock*: ``"borrowed"`` -- no timestamps.txt, ``timestamps_from: lidar``;
    ``"equal"`` -- its own timestamps.txt with the same content; ``"other"`` --
    its own clock at half the rate. *cmd* adds a channel on a faster clock.
    """
    _frames(seq, "lidar", N, 0, CLOCK)
    channels = {"lidar": {"kind": "raw", "loader": "npys"}}
    trav: dict = {"kind": "raw", "loader": "npys"}
    if trav_clock == "borrowed":
        _frames(seq, "trav", N, 10)
        trav["timestamps_from"] = "lidar"
    elif trav_clock == "equal":
        _frames(seq, "trav", N, 10, CLOCK)
    else:
        _frames(seq, "trav", N // 2, 10, CLOCK[::2])
    channels["trav"] = trav
    if cmd:
        _frames(seq, "cmd", 2 * N, 20, 100.0 + np.arange(2 * N) * 0.05)
        channels["cmd"] = {"kind": "raw", "loader": "npys"}
    write_config(seq, {"version": 1, "channels": channels})
    return seq


def _values(seq, key):
    ds = RawDataset(seq, keys=[key])
    return [float(ds[i].data[key][0]) for i in range(len(ds))], ds.timestamps[key]


# ───────────────────────────── inputs on one clock ────────────────────────────


@pytest.mark.parametrize("trav_clock", ["borrowed", "equal"])
def test_same_clock_inputs_reach_the_preprocessor_together(tmp_path, trav_clock):
    seq = _sequence(tmp_path / "seq", trav_clock=trav_clock)
    RawDataset.run_preprocess(_Sum(), seq)

    # One output per row of the shared clock, numbered by row, not by event.
    assert sorted(p.name for p in (seq / "sum").glob("*.npy")) == [
        f"{i:06d}.npy" for i in range(N)
    ]
    values, clock = _values(seq, "sum")
    assert values == [10.0 + 2 * i for i in range(N)]
    np.testing.assert_allclose(clock, CLOCK)


def test_the_output_is_registered_with_its_provenance(tmp_path):
    seq = _sequence(tmp_path / "seq")
    RawDataset.run_preprocess(_Sum(), seq)
    entry = read_config(seq)["channels"]["sum"]
    assert entry["kind"] == "preprocess"
    assert entry["timestamps_from"] == "lidar"
    assert entry["sources"] == ["lidar", "trav"]


def test_multi_output_preprocessor_with_two_inputs(tmp_path):
    seq = _sequence(tmp_path / "seq")
    RawDataset.run_preprocess(_SumAndDiff(), seq)
    assert _values(seq, "plus")[0] == [10.0 + 2 * i for i in range(N)]
    assert _values(seq, "minus")[0] == [10.0] * N


def test_sequence_preprocessors_see_zipped_frames(tmp_path):
    seq = _sequence(tmp_path / "seq")
    RawDataset.run_preprocess(_StackedSum(), seq)
    stacked = np.load(seq / "stacked" / "stacked.npy")
    assert stacked.shape == (N, 3)
    np.testing.assert_allclose(stacked[:, 0], [10.0 + 2 * i for i in range(N)])
    np.testing.assert_allclose(np.loadtxt(seq / "stacked" / "timestamps.txt"), CLOCK)

    RawDataset.run_preprocess(_PerFrameSum(), seq)
    assert _values(seq, "rowsum")[0] == [10.0 + 2 * i for i in range(N)]


def test_a_root_is_processed_sequence_by_sequence(tmp_path):
    root = tmp_path / "root"
    _sequence(root / "seq_a")
    _sequence(root / "seq_b", trav_clock="equal")
    RawDataset.run_preprocess(_Sum(), root)
    for name in ("seq_a", "seq_b"):
        values, clock = _values(root / name, "sum")
        assert values == [10.0 + 2 * i for i in range(N)]
        np.testing.assert_allclose(clock, CLOCK)
        assert "sum" in read_config(root / name)["channels"]


def test_a_single_input_run_is_unchanged(tmp_path):
    class _Double(FramePreprocessor):
        output_key = "double"
        output_loader = "npys"
        input_keys = ["lidar"]

        def __call__(self, sample: Sample) -> np.ndarray:
            return 2 * sample.data["lidar"]

    seq = _sequence(tmp_path / "seq", cmd=True)
    RawDataset.run_preprocess(_Double(), seq)
    assert _values(seq, "double")[0] == [2.0 * i for i in range(N)]


# ──────────────────────────── overwrite protection ───────────────────────────


def test_an_existing_output_is_protected(tmp_path):
    seq = _sequence(tmp_path / "seq")
    RawDataset.run_preprocess(_Sum(), seq)
    with pytest.raises(FileExistsError, match="'sum' already exists"):
        RawDataset.run_preprocess(_Sum(), seq)
    # reuse: the recipe is unchanged, so the run is a no-op.
    first = (seq / "sum" / "000000.npy").stat().st_mtime_ns
    RawDataset.run_preprocess(_Sum(), seq, reuse=True)
    assert (seq / "sum" / "000000.npy").stat().st_mtime_ns == first
    # overwrite: recomputed.
    RawDataset.run_preprocess(_Sum(), seq, overwrite=True)
    assert _values(seq, "sum")[0] == [10.0 + 2 * i for i in range(N)]


# ─────────────────────────── inputs on different clocks ──────────────────────


def test_inputs_on_different_clocks_are_refused_by_name(tmp_path):
    seq = _sequence(tmp_path / "seq", trav_clock="other")
    with pytest.raises(ValueError) as err:
        RawDataset.run_preprocess(_Sum(), seq)
    message = str(err.value)
    assert "_Sum needs its inputs together" in message
    assert "'lidar' (4 frames)" in message
    assert "'trav' (2 frames)" in message
    assert "synchronized" in message
    assert not (seq / "sum").exists()  # refused before anything is written
    assert "sum" not in read_config(seq)["channels"]


def test_one_stray_clock_among_three_inputs_is_refused(tmp_path):
    seq = _sequence(tmp_path / "seq", cmd=True)
    with pytest.raises(ValueError, match=r"'cmd' \(8 frames\)"):
        RawDataset.run_preprocess(_SumThree(), seq)
    assert not (seq / "sum3").exists()


def test_same_length_but_shifted_clocks_are_refused(tmp_path):
    """Equal counts are not a shared clock: the timestamps must be identical."""
    seq = tmp_path / "seq"
    _frames(seq, "lidar", N, 0, CLOCK)
    _frames(seq, "trav", N, 10, CLOCK + 0.03)
    write_config(
        seq,
        {
            "version": 1,
            "channels": {
                "lidar": {"kind": "raw", "loader": "npys"},
                "trav": {"kind": "raw", "loader": "npys"},
            },
        },
    )
    with pytest.raises(ValueError, match="different clocks"):
        RawDataset.run_preprocess(_Sum(), seq)
