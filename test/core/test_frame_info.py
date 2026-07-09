"""Tests for ``frame_info`` / ``frame_sequence_ids`` / ``frame_stems`` -- the
public per-frame provenance accessor on the asynchronous dataset family.
"""

import numpy as np
import pytest

from apairo import FrameRef
from apairo.core.abstract_dataset import AbstractDataset
from apairo.core.profiled_dataset import _apply_lst_filter
from apairo.core.sample import Sample
from apairo.dataset.raw import RawDataset


def _npys(seq_dir, name, frames, ts):
    d = seq_dir / name
    d.mkdir(parents=True)
    for i, fr in enumerate(frames):
        np.save(d / f"{i:06d}.npy", fr)
    np.savetxt(d / "timestamps.txt", np.asarray(ts, float))


def _npy(seq_dir, name, stacked, ts):
    d = seq_dir / name
    d.mkdir(parents=True)
    np.save(d / f"{name}.npy", stacked)
    np.savetxt(d / "timestamps.txt", np.asarray(ts, float))


def _make_seq(seq_dir, n_lidar):
    # lidar: per-frame (npys); imu: stacked (npy), a different rate -> interleaved.
    _npys(
        seq_dir,
        "lidar",
        [np.random.rand(4, 3) for _ in range(n_lidar)],
        np.linspace(0.0, 1.0, n_lidar),
    )
    n_imu = n_lidar + 2
    _npy(seq_dir, "imu", np.random.rand(n_imu, 6), np.linspace(0.05, 0.95, n_imu))


@pytest.fixture
def seq(tmp_path):
    s = tmp_path / "seq_a"
    _make_seq(s, n_lidar=3)
    RawDataset.init(s)
    return s


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "root"
    _make_seq(r / "seq_a", n_lidar=3)
    _make_seq(r / "seq_b", n_lidar=2)
    RawDataset.init(r)
    return r


def test_frame_info_matches_loaded_event_single(seq):
    ds = RawDataset(seq)
    assert len(ds) == 8  # (3 lidar + 5 imu) interleaved
    for i in range(len(ds)):
        info = ds.frame_info(i)
        assert isinstance(info, FrameRef)
        # the event's channel is the single key the async loader returns
        assert info.channel == next(iter(ds[i].data))
        # and (channel, row) recovers exactly this event's timestamp
        assert ds.timestamps[info.channel][info.row] == pytest.approx(ds[i].timestamp)
        assert info.sequence == "seq_a"


def test_frame_sequence_ids_and_stems_single(seq):
    ds = RawDataset(seq)
    assert list(ds.frame_sequence_ids) == ["seq_a"] * len(ds)
    stems = ds.frame_stems
    for i in range(len(ds)):
        info = ds.frame_info(i)
        # per-frame channel (lidar) -> file stem; stacked channel (imu) -> row.
        expected = f"{info.row:06d}"
        assert stems[i] == expected


def test_frame_info_root_carries_sequence(root):
    ds = RawDataset(root)
    seqs = set(ds.frame_sequence_ids)
    assert seqs == {"seq_a", "seq_b"}
    # Sequence boundary is consistent with frame_sequence_ids, and each event's
    # channel matches the single key the async loader returns.
    for i in range(len(ds)):
        info = ds.frame_info(i)
        assert info.sequence == ds.frame_sequence_ids[i]
        assert info.channel == next(iter(ds[i].data))


def test_frame_info_through_filter(seq):
    ds = RawDataset(seq)
    keep = [5, 2, 7]
    view = ds.filter(keep)
    for j, i in enumerate(keep):
        assert view.frame_info(j) == ds.frame_info(i)


def test_frame_info_through_select(seq):
    ds = RawDataset(seq)
    view = ds.select(["lidar", "imu"])
    for i in range(len(ds)):
        assert view.frame_info(i) == ds.frame_info(i)


def test_lst_filter_now_works_on_rawdataset(root):
    # The convergence: a (seq, stem) frame filter -- previously impossible on a
    # generic RawDataset (no frame_sequence_ids/frame_stems) -- now applies.
    ds = RawDataset(root)
    picked = {(ds.frame_sequence_ids[i], ds.frame_stems[i]) for i in (0, len(ds) - 1)}
    view = _apply_lst_filter(ds, picked)
    got = {(view.frame_sequence_ids[j], view.frame_stems[j]) for j in range(len(view))}
    assert got == picked


def test_frame_info_default_is_synchronous(tmp_path):
    # A synchronous dataset: a frame is all channels at row idx -> channel None.
    class Sync(AbstractDataset):
        def __len__(self):
            return 4

        def _load(self, idx):
            return Sample(data={"a": np.zeros(1), "b": np.ones(1)}, timestamp=None)

        @property
        def frame_sequence_ids(self):
            return np.array(["s0"] * len(self), dtype=object)

    ds = Sync()
    assert ds.frame_info(2) == FrameRef(sequence="s0", channel=None, row=2)
