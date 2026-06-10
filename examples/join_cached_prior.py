"""Two-speed dataset: expensive prior frozen in RAM, stochastic augments live.

Some features are deterministic and expensive (e.g. ground height from
cloth-simulation, normal estimation, voxel statistics). Running them every
epoch wastes CPU without any benefit.

Pattern:
  1. Register the deterministic transform on a dedicated dataset instance.
  2. select(["ground_prior"]).cache() — iterate once, pin the result in RAM.
  3. Create ds_live with only raw channels + stochastic augments.
  4. join(ds_cached) — per-index merge, same cost as reading from a dict.

After the join, each sample holds both the live augmented lidar and the
cached prior. Only the prior pass is paid up front; stochastic augments
still run every epoch.
"""

from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from apairo import Rellis3DDataset

ROOT   = Path("/data/RELLIS")
VOXEL  = 0.5   # ground-estimation grid size (metres)


def ground_height_above(pts):
    """Height of each point above the voxel-minimum ground estimate."""
    xy  = (pts[:, :2] / VOXEL).astype(np.int32)
    key = xy[:, 0] * 100_003 + xy[:, 1]
    _, inv = np.unique(key, return_inverse=True)
    cell_min = np.full(inv.max() + 1, np.inf)
    np.minimum.at(cell_min, inv, pts[:, 2])
    return (pts[:, 2] - cell_min[inv]).astype(np.float32)


def random_dropout(rate=0.05):
    def _fn(pts):
        return pts[np.random.rand(len(pts)) > rate]
    return _fn


# ---------------------------------------------------------------------------
# 1. Compute and cache the deterministic prior
#    select(["ground_prior"]) keeps only that channel in the cache so raw
#    lidar arrays are not duplicated in memory.
# ---------------------------------------------------------------------------

ds_prior = (
    Rellis3DDataset(ROOT, keys=["lidar"])
    .split("train")
    .transform("lidar", ground_height_above, output="ground_prior")
)

print("Computing ground prior (runs once)…")
ds_cached = ds_prior.select(["ground_prior"]).cache()
print(f"  cached {len(ds_cached)} frames")


# ---------------------------------------------------------------------------
# 2. Live dataset — stochastic augments re-run every epoch
# ---------------------------------------------------------------------------

ds_live = (
    Rellis3DDataset(ROOT, keys=["lidar", "trav_gt"])
    .split("train")
    .transform("lidar", random_dropout(rate=0.05))
)


# ---------------------------------------------------------------------------
# 3. Join: per-index channel merge
#    ds_live[i].data  → {"lidar": ..., "trav_gt": ...}
#    ds_cached[i].data → {"ground_prior": ...}
#    combined[i].data  → all three keys
# ---------------------------------------------------------------------------

ds_train = ds_live.join(ds_cached)

sample = ds_train[0]
print(f"lidar        : {sample.data['lidar'].shape}")
print(f"trav_gt      : {sample.data['trav_gt'].shape}")
print(f"ground_prior : {sample.data['ground_prior'].shape}")

loader = DataLoader(ds_train, batch_size=4, shuffle=True, num_workers=2)

for batch in loader:
    lidar        = batch["lidar"]         # (B, N, 4) — stochastically dropped each epoch
    trav_gt      = batch["trav_gt"]       # (B, N)
    ground_prior = batch["ground_prior"]  # (B, N)    — served from RAM, no recompute
    # ... training step ...
    break
