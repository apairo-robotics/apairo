# apairo

Unified Python loader for robotics sensor datasets — one API across synchronous and asynchronous layouts, with built-in preprocessing, filtering, and dataset composition.

All data is returned as `numpy.ndarray`. Convert to the framework of your choice.

📖 **Documentation: [apairo-robotics.github.io/apairo](https://apairo-robotics.github.io/apairo/)**

---

## Installation

```bash
pip install apairo
```

Optional extras:

```bash
pip install apairo[vision]   # Image loading (Pillow)
```

Requires Python ≥ 3.11.

---

## Quickstart

```python
import apairo

ds = apairo.SemanticKittiDataset("/data/semantic_kitti", keys=["lidar", "labels"])
sample = ds[0]
# sample.data["lidar"]   -> np.ndarray (N, 4)  float32  [x, y, z, intensity]
# sample.data["labels"]  -> np.ndarray (N,)    int64
```

---

## Supported datasets

| Class | Layout | Modalities |
|---|---|---|
| `SemanticKittiDataset` | synchronous | lidar, labels |
| `Rellis3DDataset` | synchronous | lidar, labels, poses |
| `Goose3DDataset` | synchronous | lidar, labels |
| `MNTDataset` | synchronous | lidar, labels, poses |
| `RawDataset` | asynchronous | any channels — declared in `.apairo/channels.yaml` |
| `TartanKittiDataset` | asynchronous | any TartanDrive v2 channel |

`RawDataset` is the profile-free loader for the asynchronous layout: it takes its
channels — and their format (`npy`, `npys`, `bin`, `img`, `zarr`) — entirely from
`.apairo/channels.yaml`, so it loads any such dataset, including the output of
[apairo_extractor](https://github.com/apairo-robotics/apairo_extractor), with no
code change.

---

## Command line

Installing apairo provides the `apairo` command to inspect and initialize
datasets from the terminal:

```bash
# Write/repair the .apairo sidecars by scanning a directory (root-aware, idempotent)
apairo init /data/my_dataset

# Show sequences, channels (tracked + untracked), event count and any issues
apairo status /data/my_dataset           # add --json for machine output
```

`apairo init` reconstructs the `.apairo` files for data laid out before they
existed (e.g. an older extraction) — no re-extraction needed — and the result
loads directly with `RawDataset`. See [Command Line](https://apairo-robotics.github.io/apairo/cli/) for the full reference.

---

## Pipeline

apairo provides a composable set of operations that chain together — each returns a full dataset:

```python
from apairo import Rellis3DDataset, FramePreprocessor
from torch.utils.data import DataLoader
import numpy as np

# 1. Preprocess — run once, persisted in .apairo, reloaded transparently
class TravLabel(FramePreprocessor):
    output_key = "trav_gt";  output_loader = "npys"
    input_keys = ["labels"]; timestamps_from = "lidar"; sources = ["labels"]
    def process(self, sample): return (sample.data["labels"] < 10).astype(np.uint8)

ds = Rellis3DDataset(root, keys=["lidar", "labels", "ground_height_csf"])
ds.run_preprocess(TravLabel())

# 2. Cache an expensive derived channel — computed once, served from RAM
ds.transform("ground_height_csf", expensive_smooth)
ds_prior = ds.select(["ground_height_csf"]).cache()

# 3. Build train split — filter, join cached prior, apply augmentation
valid = np.load("cache/valid_indices.npy")
ds_train = (
    Rellis3DDataset(root, keys=["lidar", "trav_gt"])
    .filter(valid)
    .join(ds_prior)
    .transform("lidar", RangeFilter(max=50.0))
)

# 4. Drop into DataLoader — no adapter needed
loader = DataLoader(ds_train, batch_size=8, shuffle=True, collate_fn=my_collate)
```

See [`examples/`](examples/) for complete runnable pipelines.

---

## Preprocessing

Define a `FramePreprocessor` or `SequencePreprocessor`, run it once — apairo persists the output and reloads it transparently on subsequent runs.

```python
from apairo.preprocess import FramePreprocessor

class TravLabel(FramePreprocessor):
    output_key      = "trav_label"
    output_loader   = "npys"
    input_keys      = ["labels"]
    timestamps_from = "labels"
    sources         = ["labels"]

    def process(self, sample) -> np.ndarray:
        return (sample.data["labels"] < 10).astype(np.uint8)

ds = apairo.Goose3DDataset("/data/goose", keys=["lidar", "labels"])
ds.run_preprocess(TravLabel())
```

See [`apairo_preprocess`](https://github.com/apairo/apairo_preprocess) for a collection of ready-made preprocessors.

---

## Transforms

Apply callables at access time — no disk writes.

```python
# Per-channel
ds.transform("lidar", RangeFilter(max=50.0))

# Sample-level — consistent mask across aligned channels
def sync_filter(sample):
    mask = np.linalg.norm(sample.data["lidar"][:, :3], axis=1) < 50.0
    sample.data["lidar"]  = sample.data["lidar"][mask]
    sample.data["labels"] = sample.data["labels"][mask]
    return sample

ds.transform(sync_filter)
```

See [`apairo_transform`](https://github.com/apairo/apairo_transform) for a collection of ready-made transforms.

---

## Filtering

`filter()` returns a dataset view restricted to frames that pass a predicate. Sweep once, persist the indices, reload without I/O cost on subsequent runs:

```python
# Compute and save
view = ds.filter("trav_gt", lambda gt: (gt == 1).sum() >= 50)
np.save("cache/valid.npy", view.indices)

# Reload — no sweep
view = ds.filter(np.load("cache/valid.npy"))
```

---

## Select & cache

`select(keys)` narrows a dataset to a subset of channels. `cache()` materialises it in RAM. Together they let you cache only the channels worth caching:

```python
ds.transform("ground_height_csf", expensive_smooth)

# Compute once, store in RAM
ds_prior = ds.select(["ground_height_csf"]).cache()

# Reuse across training runs — prior served from RAM, base channels from disk
ds_v1 = base.join(ds_prior).transform(augment_v1)
ds_v2 = base.join(ds_prior).transform(augment_v2)
```

---

## Asynchronous datasets — `synchronize()`

Asynchronous datasets (multi-rate sensor rigs) expose a timestamp-ordered event timeline: `ds[i]` is one event from one sensor. To get complete multi-channel frames, resample onto a reference clock:

```python
ds = apairo.TartanKittiDataset(seq_dir, keys=["velodyne_0", "image_left", "cmd"])

ds_sync = ds.synchronize(
    reference="velodyne_0",   # default: lowest-frequency channel
    method="latest",          # "latest" (zero-order hold) or "nearest"
    tolerance=0.05,           # drop frames with no match within ±50 ms
)

ds_sync[0].data   # {"velodyne_0": ..., "image_left": ..., "cmd": ...}
```

The result is a synchronous view — random access, shuffling, and the whole chaining API (`filter`, `select`, `cache`, `join`, `DataLoader`) work unchanged. Matching is a pure index computation; no data is read until access.

---

## Combining datasets

```python
# ConcatDataset — frame axis (different recording sessions)
combined = apairo.ConcatDataset([ds_session1, ds_session2])

# ZipDataset — channel axis (same frames, different modalities)
combined = apairo.ZipDataset(ds_base, ds_prior)
# or: ds_base.join(ds_prior)

# Built-in splits
ds_train = apairo.Rellis3DDataset(root, keys=["lidar", "labels"]).split("train")
ds_val   = apairo.Rellis3DDataset(root, keys=["lidar", "labels"]).split("val")
```

---

## Extending apairo

Add a new synchronous dataset with a YAML profile and a minimal subclass.
See [documentation](https://apairo-robotics.github.io/apairo/) for the full guide.

---

## Contributing

apairo is one repository of a small ecosystem ([apairo_transform](https://github.com/apairo-robotics/apairo_transform), [apairo_preprocess](https://github.com/apairo-robotics/apairo_preprocess), [apairo_extractor](https://github.com/apairo-robotics/apairo_extractor), [apairo_rr](https://github.com/apairo-robotics/apairo_rr)). Where a change belongs, the design invariants, and the dev workflow are documented in [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT
