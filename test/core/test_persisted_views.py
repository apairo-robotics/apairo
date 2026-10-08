"""A synchronization, frozen in ``.apairo`` and reloaded as the same frames.

``synchronize()`` recomputes its matching every session; ``persist(name)``
keeps its state -- the reference clock, one index array per channel, the
method, the tolerance and a fingerprint of every source -- and
``load_view(name)`` rebuilds the very same synchronous dataset from it, with no
data copied and no matching recomputed. A view whose sources changed since is
refused, naming the channel.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from apairo.cli import main
from apairo.core.interpolator import Interpolator
from apairo.dataset.raw import RawDataset


class Lerp(Interpolator):
    def __call__(self, t, t0, v0, t1, v1):
        a = (t - t0) / (t1 - t0)
        return (1 - a) * v0 + a * v1


class Other(Interpolator):
    def __call__(self, t, t0, v0, t1, v1):
        return v0


def _channel(seq, name, stamps):
    d = seq / name
    d.mkdir(parents=True)
    for i in range(len(stamps)):
        np.save(d / f"{i:06d}.npy", np.full(2, i, np.float64))
    np.savetxt(d / "timestamps.txt", stamps, fmt="%.6f")


def _seq(path, offset=0.0):
    """cam at 10 Hz (6 frames), imu at 100 Hz from 120 ms in, so the first
    camera frames have no IMU reading within 5 ms."""
    _channel(path, "cam", 100 + offset + np.arange(6) * 0.1)
    _channel(path, "imu", 100.12 + offset + np.arange(45) * 0.01)
    return path


def _same(a, b):
    assert len(a) == len(b)
    np.testing.assert_array_equal(a.reference_timestamps, b.reference_timestamps)
    for key in a.keys:
        np.testing.assert_array_equal(a.frame_indices[key], b.frame_indices[key])
    for i in range(len(a)):
        for key in a.keys:
            np.testing.assert_array_equal(a[i].data[key], b[i].data[key])


def test_a_persisted_view_reloads_as_the_same_frames(tmp_path):
    seq = _seq(tmp_path / "seq")
    view = RawDataset(seq).synchronize(
        reference="cam", method="nearest", tolerance=0.005
    )
    assert len(view) == 4  # frames 0 and 1 have no IMU reading within 5 ms
    path = view.persist("cam_imu")
    assert path == seq / ".apairo" / "views" / "cam_imu.npz"

    again = RawDataset(seq).load_view("cam_imu")
    _same(view, again)
    np.testing.assert_allclose(again.time_offsets("imu"), view.time_offsets("imu"))
    assert again.reference == "cam"


def test_the_registry_says_how_the_view_was_made(tmp_path):
    seq = _seq(tmp_path / "seq")
    RawDataset(seq).synchronize(
        reference="cam", method="nearest", tolerance=0.005
    ).persist("cam_imu")
    entry = yaml.safe_load((seq / ".apairo" / "views.yaml").read_text())["views"][
        "cam_imu"
    ]
    assert entry["reference"] == "cam"
    assert entry["method"] == {"cam": "nearest", "imu": "nearest"}
    assert entry["tolerance"] == 0.005
    assert entry["frames"] == 4
    assert entry["sources"]["imu"]["frames"] == 45


def test_load_view_opens_the_channels_it_needs(tmp_path):
    seq = _seq(tmp_path / "seq")
    view = RawDataset(seq).synchronize(reference="cam", method="nearest")
    view.persist("all")
    _same(view, RawDataset(seq, keys=["cam"]).load_view("all"))


def test_a_view_whose_source_changed_is_refused(tmp_path):
    seq = _seq(tmp_path / "seq")
    RawDataset(seq).synchronize(reference="cam", method="nearest").persist("v")

    np.save(seq / "imu" / "000045.npy", np.zeros(2))  # one more IMU reading
    stamps = np.loadtxt(seq / "imu" / "timestamps.txt")
    np.savetxt(
        seq / "imu" / "timestamps.txt", np.append(stamps, stamps[-1] + 0.01), fmt="%.6f"
    )
    with pytest.raises(ValueError, match="stale: channel 'imu' has 46 frames, 45 when"):
        RawDataset(seq).load_view("v")


def test_a_view_whose_clock_changed_is_refused(tmp_path):
    """Same frames, another clock: a latency declared since the view was made."""
    seq = _seq(tmp_path / "seq")
    RawDataset(seq).synchronize(reference="cam", method="nearest").persist("v")
    (seq / "apairo.yaml").write_text("version: 1\nchannels:\n  imu: {latency: 0.003}\n")
    with pytest.raises(ValueError, match="stale: the clock of channel 'imu' changed"):
        RawDataset(seq).load_view("v")


def test_a_view_is_not_overwritten_by_accident(tmp_path):
    seq = _seq(tmp_path / "seq")
    ds = RawDataset(seq)
    ds.synchronize(reference="cam", method="nearest").persist("v")
    with pytest.raises(FileExistsError, match="already persisted"):
        ds.synchronize(reference="imu", method="previous").persist("v")
    ds.synchronize(reference="imu", method="previous").persist("v", overwrite=True)
    assert RawDataset(seq).load_view("v").reference == "imu"


def test_an_unknown_view_lists_the_persisted_ones(tmp_path):
    seq = _seq(tmp_path / "seq")
    RawDataset(seq).synchronize(reference="cam").persist("a")
    with pytest.raises(KeyError, match=r"No view named 'b'.*persisted: a"):
        RawDataset(seq).load_view("b")


@pytest.mark.parametrize("name", ["", ".hidden", "a/b", "../x"])
def test_a_view_name_is_plain(tmp_path, name):
    seq = _seq(tmp_path / "seq")
    with pytest.raises(ValueError, match="A view name is"):
        RawDataset(seq).synchronize(reference="cam").persist(name)


def test_an_interpolated_channel_needs_its_interpolator_again(tmp_path):
    seq = _seq(tmp_path / "seq")
    view = RawDataset(seq).synchronize(reference="cam", method={"imu": Lerp()})
    view.persist("lerp")
    with pytest.raises(
        ValueError, match="interpolates channel 'imu' with .*Lerp: pass it again"
    ):
        RawDataset(seq).load_view("lerp")
    with pytest.raises(ValueError, match="not .*Other"):
        RawDataset(seq).load_view("lerp", interpolators={"imu": Other()})
    _same(view, RawDataset(seq).load_view("lerp", interpolators={"imu": Lerp()}))


def test_a_custom_matcher_is_not_needed_to_reload(tmp_path):
    def every_other(ts, ref):
        return np.searchsorted(ts, ref) // 2 * 2

    seq = _seq(tmp_path / "seq")
    view = RawDataset(seq).synchronize(reference="cam", method={"imu": every_other})
    view.persist("custom")
    _same(view, RawDataset(seq).load_view("custom"))


def test_an_external_clock_is_kept(tmp_path):
    seq = _seq(tmp_path / "seq")
    ticks = np.arange(100.2, 100.5, 0.05)
    view = RawDataset(seq).synchronize(reference=ticks, method="nearest")
    view.persist("ticks")
    again = RawDataset(seq).load_view("ticks")
    assert again.reference is None
    _same(view, again)


def test_a_root_persists_and_reloads_each_sequence(tmp_path):
    root = tmp_path / "root"
    _seq(root / "a")
    _seq(root / "b", offset=1000.0)
    view = RawDataset(root).synchronize(
        reference="cam", method="nearest", tolerance=0.005
    )
    view.persist("cam_imu")
    for seq in ("a", "b"):
        assert (root / seq / ".apairo" / "views" / "cam_imu.npz").is_file()
    again = RawDataset(root).load_view("cam_imu")
    assert len(again) == len(view) == 8
    for i in range(len(view)):
        np.testing.assert_array_equal(again[i].data["imu"], view[i].data["imu"])


def test_only_a_view_of_a_sequence_is_persisted():
    """Its indices address the sequence's channels: a view made over another
    asynchronous dataset has nowhere to be reloaded from."""
    from apairo.core import AbstractDataset
    from apairo.core.synchronized_view import SynchronizedView

    class InMemory(AbstractDataset):
        def __init__(self):
            self._keys = ["a", "b"]
            self.timestamps = {"a": np.arange(3.0), "b": np.arange(6) * 0.5}
            self.loaders = {"a": [0, 1, 2], "b": list(range(6))}

        def __len__(self):
            return 9

        def _load(self, idx):
            raise NotImplementedError

    view = SynchronizedView(InMemory(), reference="a", method="nearest")
    with pytest.raises(
        TypeError, match="Only a view synchronized directly on a sequence"
    ):
        view.persist("v")


def test_status_lists_the_views_and_check_their_files(tmp_path, capsys):
    seq = _seq(tmp_path / "seq")
    RawDataset(seq).synchronize(
        reference="cam", method="nearest", tolerance=0.005
    ).persist("cam_imu")
    try:
        main(["status", str(seq)])
    except SystemExit:
        pass
    assert "views       cam_imu  (4 frames on cam; cam, imu)" in capsys.readouterr().out

    (seq / ".apairo" / "views" / "cam_imu.npz").unlink()
    try:
        code = main(["check", str(seq), "--json"])
    except SystemExit as exc:
        code = exc.code
    assert code == 1
    issues = json.loads(capsys.readouterr().out)["issues"]
    assert any(
        "view 'cam_imu': index file 'views/cam_imu.npz' not found" in i for i in issues
    )


def test_reloading_does_not_match_again(tmp_path, monkeypatch):
    """The indices are read back, not recomputed: the matching never runs."""
    from apairo.core.synchronized_view import SynchronizedView

    seq = _seq(tmp_path / "seq")
    view = RawDataset(seq).synchronize(
        reference="cam", method="nearest", tolerance=0.005
    )
    view.persist("v")

    def no_matching(*args, **kwargs):
        raise AssertionError("load_view recomputed the matching")

    monkeypatch.setattr(SynchronizedView, "_match", staticmethod(no_matching))
    _same(view, RawDataset(seq).load_view("v"))
