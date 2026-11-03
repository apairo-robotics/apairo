"""A frame preprocessor that reads a temporal window of frames.

``window_size > 1`` makes the runner read each frame through its causal window
(``AbstractDataset.window(..., boundary="clip")``): one output per frame, placed
and stamped as that frame, windows clipped at a sequence start and never
crossing into another sequence. The run builds them from a sliding queue,
reading each frame once, and refuses a split. The lazy preview
``p.windowed(ds).transform(p)`` sees the same windows.
"""

from __future__ import annotations

import shutil
import weakref
from pathlib import Path

import numpy as np
import pytest

from apairo.core.config import read_config, write_config
from apairo.core.preprocessor import FramePreprocessor
from apairo.core.sample import Sample
from apairo.dataset import Rellis3DDataset
from apairo.dataset.raw import RawDataset
from apairo.preprocess.runner import _group_same_clock, _recipe_key, _sliding_windows

# Sequence name -> number of frames. Frame i of a sequence holds BASE[name] + i,
# so a value names its frame and its sequence.
LENGTHS = {"seq_a": 5, "seq_b": 4}
BASE = {"seq_a": 0.0, "seq_b": 100.0}


def _clock(name: str) -> np.ndarray:
    return 10.0 + BASE[name] + np.arange(LENGTHS[name]) * 0.1


def _root(path: Path, *, trav: bool = False) -> Path:
    """A RawDataset root of two sequences; *trav* adds a second channel on the
    lidar's clock (row i holds 1000 + the lidar value)."""
    for name, n in LENGTHS.items():
        seq = path / name
        channels: dict = {"lidar": {"kind": "raw", "loader": "npys"}}
        (seq / "lidar").mkdir(parents=True)
        for i in range(n):
            np.save(seq / "lidar" / f"{i:06d}.npy", np.full(3, BASE[name] + i))
        np.savetxt(seq / "lidar" / "timestamps.txt", _clock(name))
        if trav:
            (seq / "trav").mkdir()
            for i in range(n):
                np.save(
                    seq / "trav" / f"{i:06d}.npy", np.full(3, 1000 + BASE[name] + i)
                )
            channels["trav"] = {
                "kind": "raw",
                "loader": "npys",
                "timestamps_from": "lidar",
            }
        write_config(seq, {"version": 1, "channels": channels})
    return path


def _members(samples: list[Sample]) -> Sample:
    """Reduce a window to the values of its members, oldest -> newest."""
    anchor = samples[-1]
    data = dict(anchor.data)
    data["lidar"] = np.array([float(s.data["lidar"][0]) for s in samples])
    return Sample(data=data, timestamp=anchor.timestamp)


class _Window(FramePreprocessor):
    """Outputs the values of the frames in its window."""

    output_key = "win"
    output_loader = "npys"
    input_keys = ["lidar"]
    timestamps_from = "lidar"
    window_size = 3

    def window_reduce(self, samples):
        return _members(samples)

    def __call__(self, sample: Sample) -> np.ndarray:
        return sample.data["lidar"]


def _expected(name: str, i: int, size: int = 3, stride: int = 1) -> list[float]:
    rows = [i - k * stride for k in range(size) if i - k * stride >= 0][::-1]
    return [BASE[name] + r for r in rows]


def _outputs(seq: Path, key: str = "win") -> list[list[float]]:
    ds = RawDataset(seq, keys=[key])
    return [ds[i].data[key].tolist() for i in range(len(ds))]


# --------------------------------------------------------------- materialised


def test_one_output_per_frame_stamped_as_the_frame(tmp_path):
    root = _root(tmp_path / "root")
    RawDataset.run_preprocess(_Window(), root)
    for name, n in LENGTHS.items():
        out = root / name / "win"
        assert sorted(p.name for p in out.glob("*.npy")) == [
            f"{i:06d}.npy" for i in range(n)
        ]
        np.testing.assert_allclose(np.loadtxt(out / "timestamps.txt"), _clock(name))
        assert "win" in read_config(root / name)["channels"]


@pytest.mark.parametrize("stride", [1, 2])
def test_window_holds_past_frames_clipped_per_sequence(tmp_path, stride):
    p = _Window()
    p.window_stride = stride
    root = _root(tmp_path / "root")
    RawDataset.run_preprocess(p, root)
    for name, n in LENGTHS.items():
        # seq_b's first windows hold only seq_b values: no leak from seq_a.
        assert _outputs(root / name) == [
            _expected(name, i, stride=stride) for i in range(n)
        ]


class _Pair(_Window):
    input_keys = ["lidar", "trav"]

    def window_reduce(self, samples):
        reduced = _members(samples)
        reduced.data["trav"] = np.array([float(s.data["trav"][0]) for s in samples])
        return reduced

    def __call__(self, sample):
        return sample.data["trav"] - sample.data["lidar"]


def test_same_clock_inputs_are_windowed_per_sequence(tmp_path):
    root = _root(tmp_path / "root", trav=True)
    RawDataset.run_preprocess(_Pair(), root)
    for name, n in LENGTHS.items():
        assert _outputs(root / name) == [
            [1000.0] * len(_expected(name, i)) for i in range(n)
        ]

    # The window content itself, through the lazy view over the zipped rows.
    lazy = RawDataset(root, keys=["lidar"])
    p = _Window()
    view = p.windowed(lazy).transform(p)
    got = [view[i].data["win"].tolist() for i in range(len(view))]
    assert got == [_expected(n, i) for n in LENGTHS for i in range(LENGTHS[n])]


def test_stacked_output_format(tmp_path):
    class _Sum(_Window):
        output_loader = "npy"

        def __call__(self, sample):
            return np.array([sample.data["lidar"].sum()])

    root = _root(tmp_path / "root")
    RawDataset.run_preprocess(_Sum(), root)
    for name, n in LENGTHS.items():
        stacked = np.load(root / name / "win" / "win.npy")
        np.testing.assert_allclose(
            stacked[:, 0], [sum(_expected(name, i)) for i in range(n)]
        )


def test_window_size_one_is_a_plain_frame_preprocessor(tmp_path):
    p = _Window()
    p.window_size = 1
    root = _root(tmp_path / "root")
    RawDataset.run_preprocess(p, root)
    assert _outputs(root / "seq_a") == [[float(i)] * 3 for i in range(5)]
    assert p.windowed(root) is root


# --------------------------------------------------------------------- recipe


def test_recipe_tracks_the_window():
    class _Plain(FramePreprocessor):
        output_key = "win"
        output_loader = "npys"
        input_keys = ["lidar"]
        timestamps_from = "lidar"

        def __call__(self, sample):
            return sample.data["lidar"]

    one = _Window()
    one.window_size = 1
    three, five = _Window(), _Window()
    five.window_size = 5
    strided = _Window()
    strided.window_stride = 2
    keys = [_recipe_key(p) for p in (one, three, five, strided)]
    assert len(set(keys)) == 4

    # A reducer assigned to the instance enters by name and scalar parameters.
    class _Reducer:
        def __init__(self, gap):
            self.gap = gap

        def __call__(self, samples):
            return _members(samples)

    a, b, c = _Window(), _Window(), _Window()
    a.window_reduce, b.window_reduce, c.window_reduce = (
        _Reducer(1),
        _Reducer(1),
        _Reducer(2),
    )
    assert _recipe_key(a) == _recipe_key(b) != _recipe_key(c)

    # window_size == 1 adds nothing: recipes recorded before windows still match.
    plain_one = _Plain()
    plain_one.window_size = 1
    assert _recipe_key(_Plain()) == _recipe_key(plain_one)


def test_reuse_regenerates_when_the_window_changes(tmp_path):
    root = _root(tmp_path / "root")
    RawDataset.run_preprocess(_Window(), root)
    first = (root / "seq_a" / "win" / "000004.npy").stat().st_mtime_ns
    RawDataset.run_preprocess(_Window(), root, reuse=True)
    assert (root / "seq_a" / "win" / "000004.npy").stat().st_mtime_ns == first

    wider = _Window()
    wider.window_size = 5
    RawDataset.run_preprocess(wider, root, reuse=True)
    assert _outputs(root / "seq_a")[4] == [0.0, 1.0, 2.0, 3.0, 4.0]


# ------------------------------------------------------- lazy == materialised


def test_lazy_preview_equals_the_materialised_output(tmp_path):
    root = _root(tmp_path / "root")
    p = _Window()
    p.window_stride = 2
    lazy = p.windowed(RawDataset(root, keys=["lidar"])).transform(p)
    RawDataset.run_preprocess(p, root)
    written = RawDataset(root, keys=["win"])
    assert len(lazy) == len(written) == sum(LENGTHS.values())
    for i in range(len(lazy)):
        np.testing.assert_array_equal(lazy[i].data["win"], written[i].data["win"])
        assert lazy[i].timestamp == written[i].timestamp


def test_transform_refuses_a_windowed_preprocessor_without_its_window(tmp_path):
    ds = RawDataset(_root(tmp_path / "root"), keys=["lidar"])
    with pytest.raises(TypeError, match=r"p\.windowed\(ds\)\.transform\(p\)"):
        ds.transform(_Window())
    other = ds.window(5, reduce=_members)
    with pytest.raises(TypeError, match="window of 3 frames"):
        other.transform(_Window())
    # A window view of the right size and stride is accepted, whoever built it.
    view = ds.window(3, reduce=_members).transform(_Window())
    assert view[2].data["win"].tolist() == [0.0, 1.0, 2.0]


def test_reduce_must_keep_the_anchor_timestamp(tmp_path):
    class _Drops(_Window):
        def window_reduce(self, samples):
            return Sample(data=_members(samples).data)

    root = _root(tmp_path / "root")
    with pytest.raises(ValueError, match="anchor"):
        RawDataset.run_preprocess(_Drops(), root)


def test_a_window_needs_a_reduce(tmp_path):
    class _NoReduce(FramePreprocessor):
        output_key = "win"
        input_keys = ["lidar"]
        window_size = 2

        def __call__(self, sample):
            return sample.data["lidar"]

    root = _root(tmp_path / "root")
    with pytest.raises(NotImplementedError, match="window_reduce"):
        RawDataset.run_preprocess(_NoReduce(), root)


# ----------------------------------------------- profiled dataset, split, seam


def _mean_z(samples: list[Sample]) -> Sample:
    anchor = samples[-1]
    data = dict(anchor.data)
    data["lidar"] = np.array([s.data["lidar"][:, 2].mean() for s in samples])
    return Sample(data=data, timestamp=anchor.timestamp)


class _MeanZ(FramePreprocessor):
    output_key = "win_z"
    input_keys = ["lidar"]
    window_size = 3

    def window_reduce(self, samples):
        return _mean_z(samples)

    def __call__(self, sample):
        return sample.data["lidar"].astype(np.float32)


@pytest.fixture
def rellis(tmp_path):
    root = tmp_path / "rellis"
    shutil.copytree(Path(__file__).parent.parent / "assets" / "mini_rellis", root)
    return root


def test_profiled_dataset_windows_match_the_lazy_view(rellis):
    ds = Rellis3DDataset(rellis, keys=["lidar"])
    z = [ds[i].data["lidar"][:, 2].mean() for i in range(len(ds))]
    seq = ds.frame_sequence_ids
    p = _MeanZ()
    lazy = p.windowed(ds).transform(p)

    Rellis3DDataset.run_preprocess(p, rellis)
    written = Rellis3DDataset(rellis, keys=["lidar", "win_z"])
    assert len(written) == len(ds) == len(lazy)
    for i in range(len(ds)):
        # Never across the seam between the two sequences.
        rows = [r for r in (i - 2, i - 1, i) if r >= 0 and seq[r] == seq[i]]
        np.testing.assert_allclose(
            written[i].data["win_z"], np.array([z[r] for r in rows], np.float32)
        )
        np.testing.assert_array_equal(written[i].data["win_z"], lazy[i].data["win_z"])
        assert written.frame_stems[i] == ds.frame_stems[i]


def _lst_line(seq: str, stem: str) -> str:
    return (
        f"{seq}/os1_cloud_node_kitti_bin/{stem}.bin "
        f"{seq}/os1_cloud_node_semantickitti_label_id/{stem}.label\n"
    )


@pytest.mark.parametrize("split", ["val", "test", "mixed"])
def test_a_windowed_run_is_refused_on_frames_within_a_sequence(rellis, split):
    if split == "mixed":
        # Every frame of 00000, but only the first of 00001.
        lines = [_lst_line("00000", f"{i:06d}") for i in range(5)]
        (rellis / "pt_test.lst").write_text(
            "".join(lines) + _lst_line("00001", "000000")
        )
        split = "test"
    with pytest.raises(ValueError, match="depend on the split"):
        Rellis3DDataset.run_preprocess(_MeanZ(), rellis, split=split)
    assert not any(rellis.rglob("win_z"))
    # A lazy view persists nothing: allowed, windows over the split's frames.
    ds = Rellis3DDataset(rellis, keys=["lidar"], split=split)
    p = _MeanZ()
    view = p.windowed(ds).transform(p)
    seq = ds.frame_sequence_ids
    assert [len(view[i].data["win_z"]) for i in range(len(view))] == [
        sum(1 for r in (i - 2, i - 1, i) if r >= 0 and seq[r] == seq[i])
        for i in range(len(ds))
    ]


def _win_files(root: Path) -> dict[str, bytes]:
    return {
        str(f.relative_to(root)): f.read_bytes()
        for d in root.rglob("win_z")
        for f in sorted(d.iterdir())
    }


@pytest.mark.parametrize("split", ["train", "whole_00001"])
def test_a_split_of_whole_sequences_equals_the_full_run(rellis, tmp_path, split):
    full = tmp_path / "full"
    shutil.copytree(rellis, full)
    if split == "whole_00001":
        lines = [_lst_line("00001", f"{i:06d}") for i in range(5)]
        (rellis / "pt_test.lst").write_text("".join(lines))
        split, seq = "test", "00001"
    else:
        seq = "00000"  # pt_train.lst lists every frame of 00000

    Rellis3DDataset.run_preprocess(_MeanZ(), rellis, split=split)
    Rellis3DDataset.run_preprocess(_MeanZ(), full)
    part, whole = _win_files(rellis), _win_files(full)
    assert part and all(seq in Path(name).parts for name in part)
    assert part == {k: v for k, v in whole.items() if seq in Path(k).parts}
    written = Rellis3DDataset(rellis, keys=["lidar", "win_z"], split=split)
    assert len(written) == 5


# ------------------------------------------------------------ sliding queue


class _Spy(RawDataset):
    """Counts loads per frame and the loaded samples still alive."""

    loads: dict[int, int]
    live = 0
    peak = 0

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        type(self).loads[idx] = type(self).loads.get(idx, 0) + 1
        type(self).live += 1
        type(self).peak = max(type(self).peak, type(self).live)
        weakref.finalize(sample, _Spy._release)
        return sample

    @staticmethod
    def _release():
        _Spy.live -= 1


@pytest.mark.parametrize(("size", "stride"), [(2, 1), (3, 1), (2, 2), (10, 1)])
def test_a_run_reads_each_frame_once_and_keeps_only_the_queue(tmp_path, size, stride):
    p = _Window()
    p.window_size, p.window_stride = size, stride
    root = _root(tmp_path / "root")
    _Spy.loads, _Spy.live, _Spy.peak = {}, 0, 0
    _Spy.run_preprocess(p, root)
    n = sum(LENGTHS.values())
    assert _Spy.loads == {i: 1 for i in range(n)}
    assert _Spy.peak <= (size - 1) * stride + 1
    for name, length in LENGTHS.items():
        assert _outputs(root / name) == [
            _expected(name, i, size, stride) for i in range(length)
        ]


def _byte_equal(a: Sample, b: Sample) -> None:
    assert a.timestamp == b.timestamp
    assert sorted(a.data) == sorted(b.data)
    for key in a.data:
        x, y = np.asarray(a.data[key]), np.asarray(b.data[key])
        assert x.dtype == y.dtype and x.shape == y.shape
        assert x.tobytes() == y.tobytes()


COMBOS = [(1, 1), (2, 1), (3, 2), (4, 3), (5, 2), (10, 1)]


@pytest.mark.parametrize(("size", "stride"), COMBOS)
def test_the_queue_builds_the_lazy_windows(tmp_path, size, stride):
    p = _Window()
    p.window_size, p.window_stride = size, stride
    root = _root(tmp_path / "root", trav=True)
    single = RawDataset(root, keys=["lidar"])
    # Two inputs on one clock: the runner's row-zipping view, windowed.
    pair = _group_same_clock(RawDataset(root, keys=["lidar", "trav"]), _Pair())
    for ds in (single, pair):
        queued = list(_sliding_windows(p, ds))
        lazy = p.windowed(ds)
        assert len(queued) == len(lazy) == len(ds)
        for i, sample in enumerate(queued):
            _byte_equal(sample, lazy[i])


@pytest.mark.parametrize(("size", "stride"), COMBOS)
def test_the_queue_builds_the_lazy_windows_on_a_profiled_root(rellis, size, stride):
    p = _MeanZ()
    p.window_size, p.window_stride = size, stride
    ds = Rellis3DDataset(rellis, keys=["lidar"])
    queued = list(_sliding_windows(p, ds))
    lazy = p.windowed(ds)
    assert len(queued) == len(lazy) == len(ds)
    for i, sample in enumerate(queued):
        _byte_equal(sample, lazy[i])
