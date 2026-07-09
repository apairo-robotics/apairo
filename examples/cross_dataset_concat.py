"""Multi-dataset training: RELLIS-3D + GOOSE-3D + SemanticKITTI.

Shows where apairo's concat / repeat primitives pay off when training across
heterogeneous platforms:

  - Per-dataset normalization applied *before* concat (intensity statistics
    differ per sensor; normalise within each source, then merge).
  - Key renaming via a sample-level transform so all three datasets expose the
    same channel names before concat takes their intersection.
  - repeat() to re-balance per-dataset contribution: upsample RELLIS ×3
    so it contributes the same number of frames as GOOSE.
  - A single DataLoader that draws uniformly from all three sources.

Prerequisite: run the per-dataset traversability preprocessors first so that
"trav_gt" (RELLIS / SemanticKITTI) and "trav_label" (GOOSE) exist on disk.
"""

from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from apairo import Rellis3DDataset, Goose3DDataset, SemanticKittiDataset

RELLIS_ROOT = Path("/data/RELLIS")
GOOSE_ROOT = Path("/data/goose/GOOSE_3D/train")
KITTI_ROOT = Path("/data/semantic_kitti")


def normalize_intensity(mean, std):
    def _fn(pts):
        pts = pts.copy()
        pts[:, 3] = (pts[:, 3] - mean) / (std + 1e-6)
        return pts

    return _fn


def rename_key(src, dst):
    def _fn(sample):
        sample.data[dst] = sample.data.pop(src)
        return sample

    return _fn


# ---------------------------------------------------------------------------
# Per-dataset views
#
# Each source is normalised with its own sensor statistics, then its
# traversability channel is renamed to "trav_gt" so ConcatDataset.keys
# (which takes the intersection) sees a consistent schema.
# ---------------------------------------------------------------------------

ds_rellis = (
    Rellis3DDataset(RELLIS_ROOT, keys=["lidar", "trav_gt"])
    .split("train")
    .transform("lidar", normalize_intensity(mean=0.28, std=0.14))
)

# GOOSE uses "trav_label" — rename before concat so the intersection includes it.
ds_goose = (
    Goose3DDataset(GOOSE_ROOT, keys=["lidar", "trav_label"])
    .transform("lidar", normalize_intensity(mean=0.52, std=0.21))
    .transform(rename_key("trav_label", "trav_gt"))
)

ds_kitti = SemanticKittiDataset(
    KITTI_ROOT, keys=["lidar", "trav_gt"], split="train"
).transform("lidar", normalize_intensity(mean=0.41, std=0.18))


# ---------------------------------------------------------------------------
# Upsample the smallest source, then concat
#
# RELLIS has ~3× fewer frames than the other two; repeat(3) brings it in line.
# With stochastic augmentation each copy produces independently-augmented samples.
# ---------------------------------------------------------------------------

ds_train = ds_rellis.repeat(3).concat(ds_goose, ds_kitti)

print(f"Total training frames : {len(ds_train)}")
print(f"Active channels       : {ds_train.keys}")

loader = DataLoader(ds_train, batch_size=8, shuffle=True, num_workers=4)

for batch in loader:
    lidar = batch["lidar"]  # (B, N, 4)
    labels = batch["trav_gt"]  # (B, N)
    # ... training step ...
    break
