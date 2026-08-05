"""Regression: `.calibration` used to read empty on every single-source view
(synchronize/filter/select/window/cache) because none of them defined or
delegated `root_dir`, so `AbstractDataset.calibration`'s
`getattr(self, "root_dir", None)` always fell through to `None`. This is why
apairo_experiments ended up hard-coding extrinsics instead of trusting
`ds.calibration.get_tf(...)` on anything but a bare `RawDataset`.
"""

import numpy as np

from apairo import RawDataset
from apairo.core.config import register_static_transform


def _t(x, y, z):
    T = np.eye(4)
    T[:3, 3] = (x, y, z)
    return T


def _mini_seq(tmp_path):
    d = tmp_path / "lidar"
    d.mkdir()
    for i in range(4):
        np.save(d / f"{i:06d}.npy", np.zeros((4, 3), dtype=np.float32))
    np.savetxt(d / "timestamps.txt", np.arange(4))
    RawDataset.init(tmp_path)
    register_static_transform(tmp_path, "os_lidar", "base_link", _t(0, 0, 1))
    return tmp_path


def _assert_resolves(ds):
    np.testing.assert_allclose(
        ds.calibration.get_tf("os_lidar", "base_link")[:3, 3], [0, 0, -1]
    )


def test_synchronized_view_resolves_calibration(tmp_path):
    root = _mini_seq(tmp_path)
    ds = RawDataset(root, keys=["lidar"]).synchronize(reference="lidar")
    _assert_resolves(ds)


def test_filtered_view_resolves_calibration(tmp_path):
    root = _mini_seq(tmp_path)
    ds = RawDataset(root, keys=["lidar"]).filter(np.array([0, 1, 2]))
    _assert_resolves(ds)


def test_channel_view_resolves_calibration(tmp_path):
    root = _mini_seq(tmp_path)
    ds = RawDataset(root, keys=["lidar"]).select(["lidar"])
    _assert_resolves(ds)


def test_window_view_resolves_calibration(tmp_path):
    root = _mini_seq(tmp_path)
    ds = RawDataset(root, keys=["lidar"]).window(
        size=2, reduce=lambda samples: samples[-1]
    )
    _assert_resolves(ds)


def test_cached_dataset_resolves_calibration(tmp_path):
    root = _mini_seq(tmp_path)
    ds = RawDataset(root, keys=["lidar"]).cache()
    _assert_resolves(ds)


def test_chained_views_still_resolve_calibration(tmp_path):
    """A realistic chain (as used by trav/apairo_preprocess pipelines):
    synchronize -> filter -> cache must each pass root_dir through."""
    root = _mini_seq(tmp_path)
    ds = (
        RawDataset(root, keys=["lidar"])
        .synchronize(reference="lidar")
        .filter(np.array([0, 1, 2]))
        .cache()
    )
    _assert_resolves(ds)


def test_view_without_root_delegates_to_none_not_a_crash():
    """A view built directly over something with no root_dir at all (e.g. a
    plain in-memory list-backed dataset in a test) must not raise -- it
    should read the same empty Calibration `AbstractDataset.calibration`
    already documents for that case."""
    from apairo.core.channel_view import ChannelView

    class _NoRootDataset:
        is_synchronous = True

        def __len__(self):
            return 0

    view = ChannelView(_NoRootDataset(), ["x"])
    assert view.root_dir is None
