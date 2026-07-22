"""Regressions from the synchronous-clock + array_file features (see the audit).

Each test is a bug the audit found: ConcatDataset misreading a clocked
synchronous dataset, stacked loaders handing out views into their cache, and the
profile clock crashing on anchorless / partially-covered loads.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from apairo.core.profiled_dataset import ProfiledDataset
from apairo.dataset import Rellis3DDataset
from apairo.loader.npy_loader import NPYLoader


class _KittiDS(ProfiledDataset):
    _profile = "semantic_kitti.yaml"


MINI = Path(__file__).parents[1] / "assets" / "mini_rellis"


@pytest.fixture
def rellis(tmp_path):
    dst = tmp_path / "rellis"
    shutil.copytree(MINI, dst)
    return str(dst)


# -- H1: ConcatDataset + the per-frame clock ------------------------------------


def test_concat_of_clocked_synchronous_datasets(rellis):
    a = Rellis3DDataset(rellis, keys=["lidar", "labels"])
    assert a.is_synchronous and a.timestamps is not None  # camera clock present
    c = a.concat(Rellis3DDataset(rellis, keys=["lidar", "labels"]))
    assert c.is_synchronous is True  # was False before the fix
    np.testing.assert_allclose(
        c.timestamps, np.concatenate([a.timestamps, a.timestamps])
    )
    assert len(c) == 2 * len(a)


def test_concat_of_views_does_not_crash(rellis):
    # concat reads child clocks via the is_synchronous protocol, not a bare
    # attribute the view wrappers don't define.
    v = Rellis3DDataset(rellis, keys=["lidar", "labels"]).select(["lidar"]).cache()
    c = v.concat(v)
    assert c.is_synchronous is True
    assert c.timestamps is None or len(c.timestamps) == len(c)


# -- H2: stacked loaders must return copies, not views into the cache -----------


def test_npy_loader_returns_a_copy(tmp_path):
    np.save(tmp_path / "poses.npy", np.zeros((3, 4)))
    ld = NPYLoader(tmp_path, file="poses.npy")
    ld[0][:] += 5.0  # mutate the returned row in place
    np.testing.assert_array_equal(ld[0], np.zeros(4))  # cache uncorrupted


def test_stacked_poses_returns_a_copy(rellis):
    ds = Rellis3DDataset(rellis, keys=["lidar", "poses"])
    before = ds[0].data["poses"].copy()
    ds[0].data["poses"] += 100.0
    np.testing.assert_array_equal(ds[0].data["poses"], before)  # unchanged on re-read


# -- H4/H5: clock crashes become graceful clockless -----------------------------


def test_anchorless_load_with_clock_source_present_is_clockless(rellis):
    ds = Rellis3DDataset(rellis, keys=["poses"])  # no per-frame anchor
    assert ds.timestamps is None
    assert ds[0].timestamp is None


def test_partial_clock_coverage_is_clockless(rellis):
    shutil.rmtree(Path(rellis) / "Rellis-3D" / "00001" / "pylon_camera_node")
    ds = Rellis3DDataset(rellis, keys=["lidar", "labels"])  # 00001 lacks the camera
    assert ds.timestamps is None
    assert len(ds) == 10


# -- H3: non-zero-padded frames must sort numerically, not lexicographically ----


def test_non_zero_padded_frames_align_with_stacked_clock(tmp_path):
    # Frames named 0..11 (not zero-padded). A lexicographic sort orders them
    # 0,1,10,11,2,... so frame '10' lands at position 2 and would be assigned
    # times.txt line 2 (and any stacked/sidecar row 2) -- a silent misalignment.
    seq = tmp_path / "sequences" / "00"
    (seq / "velodyne").mkdir(parents=True)
    (seq / "labels").mkdir(parents=True)
    n = 12
    for i in range(n):
        # encode the frame index into the cloud content to verify frame order
        np.full((4, 4), float(i), np.float32).tofile(seq / "velodyne" / f"{i}.bin")
        np.zeros(4, np.int32).tofile(seq / "labels" / f"{i}.label")
    (seq / "times.txt").write_text("\n".join(f"{i / 10:.1f}" for i in range(n)))

    ds = _KittiDS(str(tmp_path), keys=["lidar", "labels"])
    for k in range(n):
        assert ds[k].data["lidar"][0, 0] == pytest.approx(float(k))  # frame k, in order
        assert ds[k].timestamp == pytest.approx(k / 10)  # clock line k, aligned
