# Two-speed dataset: cached prior + live augments

**Scenario:** A ground-height prior computed from lidar (voxel-min pooling, CSF, RANSAC…) is deterministic and expensive — ~100 ms per frame.  Re-running it every epoch is wasteful.

**Pattern:** compute it once, freeze it in RAM with `select().cache()`, then `join()` it with the live dataset so stochastic augmentations still run every epoch.

**APIs:** `transform()`, `select()`, `cache()`, `join()`.

---

## Compute and cache the prior

Register the deterministic transform, then `select()` to keep only the output channel in the cache (no raw lidar duplication in memory).  `cache()` iterates the dataset once at call time.

```python
import numpy as np
from apairo import Rellis3DDataset

VOXEL = 0.5

def ground_height_above(pts):
    xy  = (pts[:, :2] / VOXEL).astype(np.int32)
    key = xy[:, 0] * 100_003 + xy[:, 1]
    _, inv = np.unique(key, return_inverse=True)
    cell_min = np.full(inv.max() + 1, np.inf)
    np.minimum.at(cell_min, inv, pts[:, 2])
    return (pts[:, 2] - cell_min[inv]).astype(np.float32)

ds_prior = (
    Rellis3DDataset("/data/RELLIS", keys=["lidar"])
    .split("train")
    .transform("lidar", ground_height_above, output="ground_prior")
)

ds_cached = ds_prior.select(["ground_prior"]).cache()
```

`select(["ground_prior"])` projects each sample to that channel only before caching, so the raw lidar arrays are not kept in RAM.

## Live dataset with stochastic augments

Create a separate instance for the raw channels.  Stochastic transforms registered here re-run every epoch — they are **not** cached.

```python
def random_dropout(rate=0.05):
    def _fn(pts):
        return pts[np.random.rand(len(pts)) > rate]
    return _fn

ds_live = (
    Rellis3DDataset("/data/RELLIS", keys=["lidar", "trav_gt"])
    .split("train")
    .transform("lidar", random_dropout(rate=0.05))
)
```

## Join: per-index channel merge

`join()` merges at access time.  Both sides must have the same length (same split, same root).

```python
ds_train = ds_live.join(ds_cached)
# ds_train[i].data == {"lidar": ..., "trav_gt": ..., "ground_prior": ...}
```

```python
from torch.utils.data import DataLoader

loader = DataLoader(ds_train, batch_size=4, shuffle=True, num_workers=2)

for batch in loader:
    lidar        = batch["lidar"]         # stochastically dropped each epoch
    trav_gt      = batch["trav_gt"]
    ground_prior = batch["ground_prior"]  # served from RAM, no recompute
```

## Cost profile

| Step | Runs | Cost |
|---|---|---|
| `ds_cached = …select(…).cache()` | Once at startup | Full deterministic pass |
| `random_dropout` in `ds_live` | Every epoch, every frame | Negligible (numpy mask) |
| `ground_prior` lookup in `ds_cached` | Every epoch, every frame | Single dict read |

The up-front cost is paid once per training run, not once per epoch.
