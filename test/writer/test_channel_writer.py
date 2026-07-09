"""Tests for ``apairo.ChannelWriter`` -- writing a per-frame channel that the
loader reads back, with the naming/timestamps/registration policy owned by apairo.
"""

import numpy as np
import pytest

import apairo
from apairo.core.config import read_config
from apairo.loader import NPYSLoader
from apairo.writer import ChannelWriter


def _read_ts(cdir):
    return np.atleast_1d(np.loadtxt(cdir / "timestamps.txt")).tolist()


def test_writes_frames_timestamps_and_registers(tmp_path):
    seq = tmp_path / "seq"
    with ChannelWriter(
        seq, "ground_truth", loader="npys", timestamps_from="cloud", sources=["cloud"]
    ) as w:
        w.add(np.arange(5, dtype=np.int32), stem="001813", timestamp=10.0)

    cdir = seq / "ground_truth"
    assert (cdir / "001813.npy").exists()
    assert _read_ts(cdir) == [10.0]

    entry = read_config(seq)["channels"]["ground_truth"]
    assert entry["loader"] == "npys"
    assert entry["kind"] == "preprocess"
    assert entry["timestamps_from"] == "cloud"
    assert entry["sources"] == ["cloud"]


def test_written_channel_is_loadable_and_preserves_dtype(tmp_path):
    seq = tmp_path / "seq"
    labels = np.array([3, 1, 4, 1, 5], dtype=np.int32)
    with ChannelWriter(seq, "gt") as w:  # no timestamps
        w.add(labels, stem="000042")

    loader = NPYSLoader(str(seq / "gt"))
    assert len(loader) == 1
    np.testing.assert_array_equal(loader[0], labels)
    assert loader[0].dtype == np.int32
    assert not (seq / "gt" / "timestamps.txt").exists()  # none given -> none written


def test_timestamps_follow_sorted_frame_order(tmp_path):
    # Added out of order; timestamps.txt must match the loader's sorted order.
    seq = tmp_path / "seq"
    with ChannelWriter(seq, "gt") as w:
        w.add(np.zeros(2), stem="000002", timestamp=200.0)
        w.add(np.zeros(2), stem="000001", timestamp=100.0)

    cdir = seq / "gt"
    assert sorted(p.name for p in cdir.glob("*.npy")) == ["000001.npy", "000002.npy"]
    assert _read_ts(cdir) == [100.0, 200.0]  # ordered like the frames


def test_rejects_underscore_stem(tmp_path):
    seq = tmp_path / "seq"
    w = ChannelWriter(seq, "gt")
    with pytest.raises(ValueError, match="must not contain '_'"):
        w.add(np.zeros(3), stem="001813_toaster")
    assert not (seq / "gt").exists()  # nothing written on rejection


def test_resumes_existing_channel_across_instances(tmp_path):
    seq = tmp_path / "seq"
    with ChannelWriter(seq, "gt") as w:
        w.add(np.zeros(2), stem="000001", timestamp=1.0)
    # A fresh writer on the same dir must pick up the existing frame.
    with ChannelWriter(seq, "gt") as w:
        w.add(np.zeros(2), stem="000002", timestamp=2.0)

    cdir = seq / "gt"
    assert sorted(p.name for p in cdir.glob("*.npy")) == ["000001.npy", "000002.npy"]
    assert _read_ts(cdir) == [1.0, 2.0]


def test_partial_timestamps_raise_on_close(tmp_path):
    seq = tmp_path / "seq"
    w = ChannelWriter(seq, "gt")
    w.add(np.zeros(2), stem="000001", timestamp=1.0)
    w.add(np.zeros(2), stem="000002")  # no timestamp
    with pytest.raises(ValueError, match="all frames or none"):
        w.close()


def test_rejects_non_per_frame_loader(tmp_path):
    with pytest.raises(ValueError, match="per-frame loaders"):
        ChannelWriter(tmp_path / "seq", "gt", loader="npy")  # stacked, out of scope


def test_roundtrip_loads_with_rawdataset(tmp_path):
    # A sequence with a source channel + a written eval channel both load.
    seq = tmp_path / "seq"
    (seq / "cloud").mkdir(parents=True)
    ts = [100.0, 101.0, 102.0]
    for i, _t in enumerate(ts):
        np.save(seq / "cloud" / f"{i:06d}.npy", np.random.rand(8, 3))
    np.savetxt(seq / "cloud" / "timestamps.txt", np.array(ts))
    apairo.RawDataset.init(seq)  # registers the raw 'cloud' channel

    with apairo.ChannelWriter(
        seq, "ground_truth", loader="npys", timestamps_from="cloud", sources=["cloud"]
    ) as w:
        w.add(np.arange(8, dtype=np.int32), stem="000001", timestamp=ts[1])

    ds = apairo.RawDataset(seq)
    assert "ground_truth" in ds.available
    assert "cloud" in ds.available
