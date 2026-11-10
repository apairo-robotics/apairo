"""A synchronous dataset pairs its channels by position: the per-frame files
of every channel must name the same frames, not merely be as many."""

from __future__ import annotations

import numpy as np
import pytest

from apairo.core.profiled_dataset import _check_same_frames
from apairo.dataset.semantic_kitti import SemanticKittiDataset


def _tree(root, lidar_ids, label_ids, seq="00"):
    lidar, labels = (
        root / "sequences" / seq / "velodyne",
        root / "sequences" / seq / "labels",
    )
    lidar.mkdir(parents=True)
    labels.mkdir(parents=True)
    for i in lidar_ids:
        np.full((2, 4), float(i), dtype=np.float32).tofile(lidar / f"{i:06d}.bin")
    for i in label_ids:
        np.full(2, i, dtype=np.int32).tofile(labels / f"{i:06d}.label")
    return root


def test_matching_frames_load(tmp_path):
    ds = SemanticKittiDataset(
        _tree(tmp_path, [0, 1, 2], [0, 1, 2]), keys=["lidar", "labels"]
    )
    assert [int(ds[i].data["labels"][0]) for i in range(3)] == [0, 1, 2]


def test_same_count_but_shifted_frames_is_refused(tmp_path):
    root = _tree(tmp_path, [0, 1, 2], [1, 2, 3])
    with pytest.raises(ValueError, match=r"not the same frames.*'000000'.*'000001'"):
        SemanticKittiDataset(root, keys=["lidar", "labels"])


def test_each_channel_still_loads_alone(tmp_path):
    root = _tree(tmp_path, [0, 1, 2], [1, 2, 3])
    assert len(SemanticKittiDataset(root, keys=["lidar"])) == 3
    assert len(SemanticKittiDataset(root, keys=["labels"])) == 3


def test_a_constant_suffix_per_channel_is_set_aside():
    lidar = [("s", "run__0528_166_vls128"), ("s", "run__0529_167_vls128")]
    labels = [("s", "run__0528_166_goose"), ("s", "run__0529_167_goose")]
    _check_same_frames({"lidar": lidar, "labels": labels}, "root")
    shifted = [("s", "run__0529_167_goose"), ("s", "run__0530_168_goose")]
    with pytest.raises(ValueError, match="not the same frames"):
        _check_same_frames({"lidar": lidar, "labels": shifted}, "root")


def test_unrelated_naming_schemes_are_paired_by_position():
    lidar = [("s", "000104"), ("s", "000105")]
    image = [("s", "frame000104-1581624663_149"), ("s", "frame000105-1581624663_249")]
    _check_same_frames({"lidar": lidar, "image": image}, "root")


def test_frames_of_another_sequence_are_refused():
    a = [("00", "000000"), ("01", "000000")]
    b = [("00", "000000"), ("02", "000000")]
    with pytest.raises(ValueError, match="sequence '01'.*'02'"):
        _check_same_frames({"lidar": a, "labels": b}, "root")
