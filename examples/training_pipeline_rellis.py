"""Traversability training pipeline on RELLIS-3D using apairo.

The goal of this example is to show where apairo brings value:
  - Preprocessing  : define once, persist to disk, reload transparently next run
  - Splits         : train / val / test from the dataset's built-in LST files
  - Transforms     : deterministic ops cached, stochastic ops applied after
  - DataLoader     : ds_train / ds_val plug directly into PyTorch — no adapter needed

The .cache() call is the explicit boundary between deterministic and stochastic:
everything before it is frozen in RAM, everything after runs fresh every access.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree
from torch.utils.data import DataLoader

from apairo import Rellis3DDataset, FramePreprocessor, SequencePreprocessor
from apairo.core.sample import Sample


# ---------------------------------------------------------------------------
# 1. Preprocessors — declared once, run once, stored in .apairo
#    Next run: apairo detects existing outputs and skips recomputation.
# ---------------------------------------------------------------------------

_RELLIS_TRAVERSABLE_IDS = {1, 3, 10, 23, 31, 33}


class TraversabilityFromLabels(FramePreprocessor):
    """Ground-truth traversability from semantic label IDs (one file per frame)."""

    output_key      = "trav_gt"
    output_loader   = "npys"
    input_keys      = ["labels"]
    timestamps_from = "lidar"
    sources         = ["labels"]

    def process(self, sample: Sample) -> np.ndarray:
        return np.isin(sample.data["labels"], list(_RELLIS_TRAVERSABLE_IDS)).astype(np.uint8)


class TraversabilityFromTrajectory(SequencePreprocessor):
    """Observed traversability: points the robot actually drove through."""

    output_key      = "trav_obs"
    output_loader   = "npys"
    input_keys      = ["lidar", "poses"]
    timestamps_from = "lidar"
    sources         = ["lidar", "poses"]

    def __init__(self, robot_radius: float = 0.75) -> None:
        self._radius = robot_radius

    def process(self, frames) -> np.ndarray:
        samples   = list(frames)
        poses     = np.stack([s.data["poses"] for s in samples]).astype(np.float64)
        positions = poses[:, :3, 3]

        results = []
        for i, sample in enumerate(samples):
            pc       = np.asarray(sample.data["lidar"])[:, :3].astype(np.float64)
            pc_world = (poses[i, :3, :3] @ pc.T).T + positions[i]
            future   = positions[i + 1:]
            if len(future) == 0:
                results.append(np.zeros(len(pc), dtype=np.uint8))
                continue
            dist_xy, _ = KDTree(future[:, :2]).query(pc_world[:, :2], k=1)
            results.append((dist_xy < self._radius).astype(np.uint8))

        return np.stack(results)


# ---------------------------------------------------------------------------
# 2. Transforms
# ---------------------------------------------------------------------------

class RobotFilter:
    """Remove points within *d* metres of the sensor (robot body). Deterministic."""

    def __init__(self, d: float = 1.0) -> None:
        self._d = d

    def compute_mask(self, pc: np.ndarray) -> np.ndarray:
        return np.max(np.abs(pc[:, :3]), axis=1) >= self._d

    def __call__(self, pc: np.ndarray) -> np.ndarray:
        return pc[self.compute_mask(pc)]


def random_subsample(n: int):
    """Subsample lidar and trav_obs to the same *n* random indices. Stochastic."""
    def _fn(sample: Sample) -> Sample:
        pc  = sample.data["lidar"]
        idx = np.random.choice(len(pc), size=n, replace=len(pc) < n)
        sample.data["lidar"]    = pc[idx]
        sample.data["trav_obs"] = sample.data["trav_obs"][idx]
        return sample
    return _fn


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

root = Path(os.environ.get("APAIRO_RELLIS_ROOT", "/data/RELLIS"))

# Preprocess — run once, results stored in .apairo alongside the raw data
ds_pre = Rellis3DDataset(root, keys=["lidar", "poses", "labels"])
ds_pre.run_preprocess(TraversabilityFromLabels())
ds_pre.run_preprocess(TraversabilityFromTrajectory())

# Splits — apairo reads RELLIS's built-in pt_train/val/test.lst files
ds = Rellis3DDataset(root, keys=["lidar", "trav_obs"])

# Deterministic transforms before .cache() — computed once, frozen in RAM.
# Stochastic transforms after .cache()   — run fresh every access.
ds_train_base = (
    ds.split("train")
    .transform("lidar", RobotFilter(d=1.0))   # deterministic
    .cache()                                   # <-- boundary
)
ds_val_base = (
    ds.split("val")
    .transform("lidar", RobotFilter(d=1.0))   # deterministic
    .cache()                                   # <-- boundary
)

ds_train = ds_train_base.transform(random_subsample(n=4096))  # stochastic
ds_val   = ds_val_base.transform(random_subsample(n=4096))    # stochastic

# Training — ds_train / ds_val are standard PyTorch datasets. Iterate them with
# your own collate_fn (point clouds are ragged, so the default collate can't
# stack them — pick padding or a list; see the README), then hand the loaders to
# your trainer:  trainer = Trainer(model, train_loader, val_loader); trainer.fit()
train_loader = DataLoader(ds_train, batch_size=4, shuffle=True)
val_loader   = DataLoader(ds_val,   batch_size=4, shuffle=False)
print(f"train frames: {len(ds_train)}  |  val frames: {len(ds_val)}")
