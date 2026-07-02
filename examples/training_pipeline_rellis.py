"""Traversability training pipeline on RELLIS-3D using apairo.

Shows where apairo brings value:
  - Preprocessing : define once, persist to .apairo, reload transparently.
  - Splits        : train / val / test from the dataset's built-in LST files.
  - Transforms    : deterministic ops cached, stochastic ops applied after.

The .cache() call is the explicit boundary between deterministic and stochastic:
everything before it is frozen in RAM, everything after runs fresh every access.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from apairo import Rellis3DDataset, FramePreprocessor
from apairo.core.sample import Sample


# ---------------------------------------------------------------------------
# 1. Preprocessor — declared once, run once, stored in .apairo. A re-run raises
#    FileExistsError (output already there); pass overwrite=True to recompute.
# ---------------------------------------------------------------------------

_RELLIS_TRAVERSABLE_IDS = [1, 3, 10, 23, 31, 33]


class TraversabilityFromLabels(FramePreprocessor):
    """Ground-truth traversability from semantic label IDs (one file per frame)."""

    output_key      = "trav_gt"
    output_loader   = "npys"
    input_keys      = ["labels"]
    timestamps_from = "lidar"
    sources         = ["labels"]

    def __call__(self, sample: Sample) -> np.ndarray:
        return np.isin(sample.data["labels"], _RELLIS_TRAVERSABLE_IDS).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2. Transforms
# ---------------------------------------------------------------------------

class RobotFilter:
    """Drop points within *d* metres of the sensor, consistently across the
    aligned lidar + trav_gt channels.  Deterministic -- safe before .cache()."""

    def __init__(self, d: float = 1.0) -> None:
        self._d = d

    def __call__(self, sample: Sample) -> Sample:
        pts  = sample.data["lidar"]
        keep = np.max(np.abs(pts[:, :3]), axis=1) >= self._d
        sample.data["lidar"]   = pts[keep]
        sample.data["trav_gt"] = sample.data["trav_gt"][keep]
        return sample


def random_subsample(n: int):
    """Subsample lidar and trav_gt to the same *n* random indices. Stochastic."""
    def _fn(sample: Sample) -> Sample:
        pc  = sample.data["lidar"]
        idx = np.random.choice(len(pc), size=n, replace=len(pc) < n)
        sample.data["lidar"]   = pc[idx]
        sample.data["trav_gt"] = sample.data["trav_gt"][idx]
        return sample
    return _fn


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

root = Path(os.environ.get("APAIRO_RELLIS_ROOT", "/data/RELLIS"))

# Preprocess — run once, persisted in .apairo (FileExistsError = already done).
try:
    Rellis3DDataset(root, keys=["lidar", "labels"]).run_preprocess(TraversabilityFromLabels())
except FileExistsError:
    pass

# Splits — apairo reads RELLIS's built-in pt_train/val/test.lst files.
# filter_split() applies the split to the loaded dataset (keeping the derived
# trav_gt channel) and chains mid-pipeline.
ds = Rellis3DDataset(root, keys=["lidar", "trav_gt"])

# Deterministic transforms before .cache() — computed once, frozen in RAM.
# Stochastic transforms after .cache()   — run fresh every access.
ds_train_base = ds.filter_split("train").transform(RobotFilter(d=1.0)).cache()  # <-- boundary
ds_val_base   = ds.filter_split("val").transform(RobotFilter(d=1.0)).cache()    # <-- boundary

ds_train = ds_train_base.transform(random_subsample(n=4096))  # stochastic
ds_val   = ds_val_base.transform(random_subsample(n=4096))    # stochastic

print(f"train frames: {len(ds_train)}  |  val frames: {len(ds_val)}")

# ds_train / ds_val are standard PyTorch datasets. Wrap each in a torch DataLoader
# with your own collate_fn — point clouds are ragged, so the default collate can't
# stack them (pick padding or a list; see the README) — then hand the loaders to
# your trainer:  trainer = Trainer(model, train_loader, val_loader); trainer.fit()
