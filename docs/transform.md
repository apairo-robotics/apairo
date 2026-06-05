# Transforms

Transforms let you apply callables to channel data **at access time**, without persisting anything to disk. This is the right tool for normalisations, type conversions, augmentations, or any operation that is cheap enough to run on the fly.

> **Companion library:** [`apairo_transform`](https://github.com/apairo/apairo_transform) ships a collection of ready-made transforms (range filters, normalisation, voxelisation, …) that plug directly into this API.

---

## `dataset.transform(key, fn)`

Register a callable on a channel. The callable receives the raw loaded value and returns the transformed value. The method returns `self` so calls chain:

```python
ds = Goose3DDataset("/data/goose", keys=["lidar", "labels"])

ds.transform("lidar", lambda pts: pts[pts[:, 2] > -2])   # ground filter
  .transform("lidar", lambda pts: pts / pts.max())        # normalise
```

Multiple calls on the same key compose **in registration order** (first registered, first applied).

---

## `Compose`

`Compose` wraps a sequence of callables into a single transform object. Useful when you want to name, reuse, or inspect a pipeline:

```python
from apairo import Compose

pipeline = Compose([
    lambda pts: pts[pts[:, 2] > -2],
    lambda pts: pts / pts.max(),
])

ds.transform("lidar", pipeline)
print(pipeline)  # Compose([<lambda>, <lambda>])
```

### With named callables

```python
class RangeFilter:
    def __init__(self, max_range: float) -> None:
        self.max_range = max_range

    def __call__(self, pts):
        dists = (pts[:, :3] ** 2).sum(axis=1) ** 0.5
        return pts[dists < self.max_range]

class ZNorm:
    def __call__(self, pts):
        return (pts - pts.mean(0)) / (pts.std(0) + 1e-6)

ds.transform("lidar", Compose([RangeFilter(max_range=50), ZNorm()]))
```

---

## Behaviour

| Property | Detail |
|---|---|
| **No disk writes** | Transforms run in memory at `__getitem__` time. |
| **Order** | Applied in the order `transform()` was called. Multiple `Compose` objects on the same key also compose in registration order. |
| **Scope** | Per-instance. Transforms registered on `ds` do not affect another instance pointing at the same path. |
| **Chaining** | `transform()` returns `self`, so it can be chained directly on the constructor result. |

---

## Using `apairo_transform`

[`apairo_transform`](https://github.com/apairo/apairo_transform) provides drop-in transforms for common LiDAR preprocessing steps:

```bash
pip install apairo_transform
```

```python
from apairo_transform import RangeFilter, VoxelDownsample, IntensityNorm

ds.transform("lidar", Compose([
    RangeFilter(max_range=50.0),
    VoxelDownsample(voxel_size=0.1),
    IntensityNorm(),
]))
```

All transforms in `apairo_transform` are plain callables -- they work with `Compose` and `dataset.transform` out of the box, with no extra glue.

---

## `dataset.sample_transform(fn)`

Use `sample_transform` when an operation must touch several channels consistently. The callable receives the full `Sample` and returns a (possibly new) `Sample`, so you control all channels atomically with no shared state.

```python
def range_filter(sample: Sample) -> Sample:
    mask = sample.data["lidar"][:, :3].norm(dim=1) < 50.0
    sample.data["lidar"]  = sample.data["lidar"][mask]
    sample.data["labels"] = sample.data["labels"][mask]
    return sample

ds.sample_transform(range_filter)
```

Multiple calls compose in registration order. `sample_transform` returns `self` for chaining, and can be mixed with per-channel `transform` calls:

```python
ds.transform("lidar", IntensityNorm())        # per-channel, runs first
  .sample_transform(range_filter)             # whole sample, runs after
```

**Execution order:** per-channel transforms always run before sample transforms, regardless of registration order.

### When to use which

| | `transform(key, fn)` | `sample_transform(fn)` |
|---|---|---|
| Scope | One channel | All channels |
| Use case | Normalisation, type cast, augmentation on a single modality | Geometry-consistent filtering, cross-channel operations |
| State | Stateless | Stateless |
