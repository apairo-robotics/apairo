# Multi-dataset training

**Scenario:** You want to train on RELLIS-3D, GOOSE-3D, and SemanticKITTI simultaneously.  Each dataset has its own sensor, so intensity statistics differ.  RELLIS is also 3× smaller than the others.

**APIs:** `concat()`, `repeat()`, per-channel `transform()`, sample-level `transform()`.

---

## Dataset-specific normalisation before concat

Apply normalisation *per source* before merging so that each sensor's intensity distribution is centred the same way.  `ConcatDataset` takes the key intersection — rename GOOSE's `trav_label` to `trav_gt` first so all three sources expose the same schema.

```python
from pathlib import Path
from torch.utils.data import DataLoader
from apairo import Rellis3DDataset, Goose3DDataset, SemanticKittiDataset

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

ds_rellis = (
    Rellis3DDataset("/data/RELLIS", keys=["lidar", "trav_gt"])
    .split("train")
    .transform("lidar", normalize_intensity(mean=0.28, std=0.14))
)

ds_goose = (
    Goose3DDataset("/data/GOOSE_3D/train", keys=["lidar", "trav_label"])
    .transform("lidar", normalize_intensity(mean=0.52, std=0.21))
    .transform(rename_key("trav_label", "trav_gt"))
)

ds_kitti = (
    SemanticKittiDataset("/data/semantic_kitti", keys=["lidar", "trav_gt"], split="train")
    .transform("lidar", normalize_intensity(mean=0.41, std=0.18))
)
```

## Upsampling with `repeat()`

RELLIS has ~3× fewer frames than the other two.  `repeat(3)` makes it contribute proportionally.  With stochastic augmentation, each repetition produces different samples.

```python
ds_train = ds_rellis.repeat(3).concat(ds_goose, ds_kitti)

print(f"Total frames : {len(ds_train)}")
print(f"Keys         : {ds_train.keys}")  # ["lidar", "trav_gt"]
```

`concat()` takes the key intersection — all three sources expose `["lidar", "trav_gt"]` after renaming.

## DataLoader

```python
loader = DataLoader(ds_train, batch_size=8, shuffle=True, num_workers=4)
```

`ds_train` is a standard PyTorch dataset; no adapter needed.

!!! note "Prerequisite"
    Run the per-dataset traversability preprocessors first so that `trav_gt` (RELLIS / SemanticKITTI) and `trav_label` (GOOSE) exist on disk.  See [Preprocessing](../preprocessing.md).
