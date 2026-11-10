"""An interrupted preprocess leaves nothing that blocks the next run, and a
sequence preprocessor reads its frames one at a time."""

from __future__ import annotations

import numpy as np
import pytest

from apairo import RawDataset
from apairo.core.preprocessor import FramePreprocessor, SequencePreprocessor
from apairo.preprocess.runner import _IN_PROGRESS

N = 6


def _seq(path):
    d = path / "lidar"
    d.mkdir(parents=True)
    for i in range(N):
        np.save(d / f"{i:06d}.npy", np.full(3, float(i)))
    np.savetxt(d / "timestamps.txt", np.arange(N) * 0.1)
    return path


class Double(FramePreprocessor):
    output_key = "double"
    input_keys = ["lidar"]
    timestamps_from = "lidar"
    stop_at: int | None = None
    stop_with: type[BaseException] = KeyboardInterrupt

    def __init__(self):
        self.calls = 0

    def __call__(self, sample):
        self.calls += 1
        if self.stop_at is not None and self.calls > self.stop_at:
            raise self.stop_with()
        return sample.data["lidar"] * 2


@pytest.mark.parametrize("error", [KeyboardInterrupt, RuntimeError])
def test_an_interrupted_run_leaves_nothing_and_runs_again(tmp_path, error):
    seq = _seq(tmp_path / "seq")
    p = Double()
    p.stop_at, p.stop_with = 3, error
    with pytest.raises(error):
        RawDataset.run_preprocess(p, seq)
    assert not (seq / "double").exists()

    RawDataset.run_preprocess(Double(), seq)  # no flag needed
    ds = RawDataset(seq, keys=["double"])
    assert len(ds) == N and not (seq / "double" / _IN_PROGRESS).exists()


def test_the_leftover_of_a_killed_run_is_removed(tmp_path):
    seq = _seq(tmp_path / "seq")
    RawDataset(seq)  # the killed run had opened the sequence: its sidecar exists
    out = seq / "double"
    out.mkdir()
    np.save(out / "000000.npy", np.zeros(3))  # half an output...
    (out / _IN_PROGRESS).touch()  # ...whose run never finished

    RawDataset.run_preprocess(Double(), seq)
    assert len(RawDataset(seq, keys=["double"])) == N


def test_data_apairo_did_not_write_is_still_left_alone(tmp_path):
    seq = _seq(tmp_path / "seq")
    RawDataset(seq)  # the sidecar exists before the foreign directory appears
    (seq / "double").mkdir()
    np.save(seq / "double" / "mine.npy", np.zeros(3))
    with pytest.raises(FileExistsError):
        RawDataset.run_preprocess(Double(), seq)
    with pytest.raises(FileExistsError, match="did not write"):
        RawDataset.run_preprocess(Double(), seq, overwrite=True)
    assert (seq / "double" / "mine.npy").exists()


def test_an_interrupted_overwrite_is_not_reused(tmp_path):
    seq = _seq(tmp_path / "seq")
    RawDataset.run_preprocess(Double(), seq)
    p = Double()
    p.stop_at = 2
    with pytest.raises(KeyboardInterrupt):
        RawDataset.run_preprocess(p, seq, overwrite=True)
    RawDataset.run_preprocess(Double(), seq, reuse=True)  # recomputed, not skipped
    assert len(RawDataset(seq, keys=["double"])) == N


def test_a_sequence_preprocessor_reads_frames_one_at_a_time(tmp_path):
    seq = _seq(tmp_path / "seq")
    alive: list[int] = []

    class Peak(SequencePreprocessor):
        output_key = "peak"
        input_keys = ["lidar"]
        timestamps_from = "lidar"

        def __call__(self, frames):
            out = []
            for n, sample in enumerate(frames):
                alive.append(n)  # frame n arrives only now, after n - 1 was used
                out.append(sample.data["lidar"] * 2)
                if n == 0:
                    assert alive == [0]  # the rest of the sequence is not loaded yet
            return np.stack(out)

    RawDataset.run_preprocess(Peak(), seq)
    assert alive == list(range(N))
    assert np.loadtxt(seq / "peak" / "timestamps.txt").shape == (N,)


def test_frames_a_sequence_preprocessor_skips_still_get_their_times(tmp_path):
    seq = _seq(tmp_path / "seq")

    class First(SequencePreprocessor):
        output_key = "first"
        input_keys = ["lidar"]
        timestamps_from = "lidar"
        output_loader = "npys"

        def __call__(self, frames):
            first = next(iter(frames)).data["lidar"]
            return np.stack([first] * N)  # one row per frame, reading only one

    RawDataset.run_preprocess(First(), seq)
    assert np.loadtxt(seq / "first" / "timestamps.txt").shape == (N,)
