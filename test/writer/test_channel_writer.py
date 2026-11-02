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
    with pytest.raises(ValueError, match="one file per frame"):
        ChannelWriter(tmp_path / "seq", "gt", loader="npy")  # stacked, out of scope


def test_rejects_a_read_only_format(tmp_path):
    with pytest.raises(ValueError, match="'pcd' is read-only"):
        ChannelWriter(tmp_path / "seq", "gt", loader="pcd")


def test_rejects_an_unknown_format(tmp_path):
    with pytest.raises(KeyError, match="Unknown loader 'h5'"):
        ChannelWriter(tmp_path / "seq", "gt", loader="h5")


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


def test_unpadded_stems_keep_their_own_timestamp(tmp_path):
    """Frames named 1, 2, 10 are read in numeric order: timestamps.txt follows
    that order, not the order of the names as text."""
    seq = tmp_path / "seq"
    with ChannelWriter(seq, "labels") as w:
        for i in (10, 2, 1, 11):
            w.add(np.full(2, float(i)), stem=str(i), timestamp=i * 0.1)
    ds = apairo.RawDataset(seq, keys=["labels"])
    values = [float(ds.loaders["labels"][i][0]) for i in range(4)]
    assert values == [1.0, 2.0, 10.0, 11.0]
    np.testing.assert_allclose(ds.timestamps["labels"], np.array(values) * 0.1)


def test_a_bin_channel_takes_n_by_4_points_only(tmp_path):
    with pytest.raises(ValueError, match=r"cannot hold.*shape \(8, 3\)"):
        ChannelWriter(tmp_path / "seq", "cloud", loader="bin").add(
            np.zeros((8, 3), np.float32), stem="000000"
        )

# ─────────────────────── any per-frame format that writes ────────────────────


def _cloud_sequence(seq, n=3):
    (seq / "cloud").mkdir(parents=True)
    for i in range(n):
        np.save(seq / "cloud" / f"{i:06d}.npy", np.random.rand(8, 3))
    np.savetxt(seq / "cloud" / "timestamps.txt", 100.0 + np.arange(n))
    apairo.RawDataset.init(seq)
    return seq


def test_img_masks_round_trip_through_rawdataset(tmp_path):
    seq = _cloud_sequence(tmp_path / "seq")
    rng = np.random.default_rng(0)
    masks = [rng.integers(0, 256, (6, 8), dtype=np.uint8) for _ in range(3)]
    with ChannelWriter(
        seq, "mask", loader="img", timestamps_from="cloud", sources=["cloud"]
    ) as w:
        for i, mask in enumerate(masks):
            w.add(mask, stem=f"{i:06d}", timestamp=100.0 + i)

    assert sorted(p.name for p in (seq / "mask").iterdir()) == [
        "000000.png",
        "000001.png",
        "000002.png",
        "timestamps.txt",
    ]
    assert read_config(seq)["channels"]["mask"]["loader"] == "img"
    ds = apairo.RawDataset(seq, keys=["mask"])
    assert len(ds) == 3
    for i, mask in enumerate(masks):
        got = ds.load("mask", i)
        assert got.dtype == np.uint8
        np.testing.assert_array_equal(got, mask)
    np.testing.assert_allclose(ds.timestamps["mask"], [100.0, 101.0, 102.0])


def test_img_resumes_its_own_frames(tmp_path):
    seq = tmp_path / "seq"
    with ChannelWriter(seq, "mask", loader="img") as w:
        w.add(np.zeros((2, 2), np.uint8), stem="000001", timestamp=1.0)
    with ChannelWriter(seq, "mask", loader="img") as w:
        w.add(np.ones((2, 2), np.uint8), stem="000002", timestamp=2.0)
    assert _read_ts(seq / "mask") == [1.0, 2.0]


@pytest.mark.parametrize(
    "loader, frame",
    [
        ("img", np.zeros((6, 8), np.float32)),  # PNG holds uint8 / uint16
        ("img", np.zeros((6, 8, 2), np.uint8)),  # neither grey, RGB nor RGBA
        ("bin", np.zeros((5, 3), np.float32)),  # bin is (N, 4) float32
        ("npys", np.array([object()], dtype=object)),
    ],
)
def test_a_frame_the_format_cannot_hold_is_refused(tmp_path, loader, frame):
    seq = tmp_path / "seq"
    w = ChannelWriter(seq, "gt", loader=loader)
    with pytest.raises(ValueError, match=f"format '{loader}' cannot hold"):
        w.add(frame, stem="000000")
    assert not (seq / "gt").exists()  # nothing written on refusal


def test_a_later_frame_the_format_cannot_hold_is_refused(tmp_path):
    seq = tmp_path / "seq"
    w = ChannelWriter(seq, "mask", loader="img")
    w.add(np.zeros((2, 2), np.uint8), stem="000000")
    with pytest.raises(ValueError, match="cannot hold"):
        w.add(np.zeros((2, 2), np.float64), stem="000001")
    assert [p.name for p in (seq / "mask").iterdir()] == ["000000.png"]


@pytest.mark.parametrize(
    "loader, frame",
    [
        ("npys", np.arange(12, dtype=np.int16).reshape(3, 4)),
        ("npys", np.asfortranarray(np.arange(12.0).reshape(3, 4))),
        ("bin", np.arange(20, dtype=np.float32).reshape(5, 4)),
    ],
)
def test_npys_and_bin_write_the_bytes_they_always_did(tmp_path, loader, frame):
    from apairo.writer import BINWriter, NPYWriter

    legacy = {"npys": (NPYWriter, ".npy"), "bin": (BINWriter, ".bin")}[loader]
    expected = tmp_path / f"expected{legacy[1]}"
    legacy[0]().write(frame, expected)
    with ChannelWriter(tmp_path / "seq", "gt", loader=loader) as w:
        w.add(frame, stem="000000")
    written = tmp_path / "seq" / "gt" / f"000000{legacy[1]}"
    assert written.read_bytes() == expected.read_bytes()
