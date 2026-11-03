# Transforms

Transforms let you apply callables to channel data **at access time**, without persisting anything to disk. This is the right tool for normalisations, type conversions, augmentations, or any operation cheap enough to run on the fly.

> **Companion library:** [`apairo_transform`](https://github.com/apairo-robotics/apairo_transform) ships a collection of ready-made transforms (range filters, normalisation, voxelisation, …) that plug directly into this API.

---

## `dataset.transform()`

A single method, three forms. All steps are registered in order and run as a unified pipeline at access time.

Use the returned dataset: `ds = ds.transform(...)`. In 0.11 `transform()` will
return a new dataset and leave the one it is called on unchanged, like
`synchronize`, `window`, `filter`, `select` and `cache` already do. See
[The returned dataset](#the-returned-dataset-in_place).

### Per-channel form

```python
ds = ds.transform(key, fn, output=None, keep=True)
```

`fn` receives `sample.data[key]` and returns the transformed value. By default the result replaces `key` in the sample:

```python
ds = (ds.transform("lidar", lambda pts: pts[pts[:, 2] > -2])
        .transform("lidar", lambda pts: pts / pts.max()))
```

### Sample-level form

```python
ds = ds.transform(fn)   # fn: Sample -> Sample
```

`fn` receives the full `Sample`. Use this when an operation must touch several channels consistently:

```python
def range_filter(sample):
    mask = sample.data["lidar"][:, :3].max(axis=1) < 50.0
    sample.data["lidar"]  = sample.data["lidar"][mask]
    sample.data["labels"] = sample.data["labels"][mask]
    return sample

ds = ds.transform(range_filter)
```

### Preprocessor form

```python
ds = ds.transform(preprocessor)   # a FramePreprocessor instance
```

A `FramePreprocessor` is a callable on a `Sample`, so it plugs into the
pipeline directly: its result is published under its declared `output_key`
(override with `output=`) at access time, nothing is written to disk. This is
the lazy preview of a preprocess — see
[Preprocessing](preprocessing.md#preview-before-materializing). A
`SequencePreprocessor` is rejected (it needs the full sequence at once).

All forms compose **in registration order**:

```python
ds = (ds.transform("lidar", Normalize())   # step 1
        .transform(range_filter)           # step 2 — sees normalised lidar
        .transform("lidar", Voxelize()))   # step 3
```

---

## Publishing a channel — `output`

Pass `output` to write the result of a per-channel transform to a **new key** while leaving the source intact. The published channel is then visible to all subsequent pipeline steps:

```python
ds = ds.transform("lidar", RangeFilter(max=50.0), output="lidar_f")

ds = ds.transform("lidar_f", Normalize(), output="lidar_norm")   # reads lidar_f
ds = ds.transform("lidar_f", Voxelize(), output="lidar_vox")     # reads lidar_f too
```

Each step reads `lidar_f` as the steps before it left it. Give each derived
result its own `output`, as above, to keep `lidar_f` intact for the steps
that follow: without `output`, `Normalize()` would replace `lidar_f` and
`Voxelize()` would read the normalised cloud.

---

## Temporary channels — `keep=False`

Set `keep=False` alongside `output` to drop the published channel from the final sample. Useful for intermediate results that are only needed within the pipeline:

```python
ds = ds.transform("lidar", compute_mask_fn, output="_mask", keep=False)
ds = ds.transform(lambda s: apply_mask(s, "_mask"))
# "_mask" is gone from the returned sample; "lidar" and "labels" are filtered
```

---

## `Compose`

Chain `transform()` calls to build a dataset. Use `Compose` to name an
operation made of several callables, so it can be reused as one, inside a
dataset or outside of it:

```python
from apairo import Compose

clean_lidar = Compose([RangeFilter(max=50.0), Normalize()])

ds = ds.transform("lidar", clean_lidar)
points = clean_lidar(raw_points)        # the same operation on a bare array
```

`Compose` is not a second way to build a pipeline: it is one step, made of
parts.

---

## Behaviour summary

| Property | Detail |
|---|---|
| **No disk writes** | Transforms run in memory at `__getitem__` time. |
| **Order** | All steps (per-channel and sample-level) run in registration order. |
| **Scope** | Per-instance. Transforms on `ds` do not affect another instance at the same path. |
| **`output`** | Publishes result as a new channel; source channel unchanged. |
| **`keep=False`** | Removes an `output` channel from the final sample after the full pipeline runs. |
| **`in_place`** | `False` returns an independent copy, `True` modifies the dataset. Unset: as `True`, with a `FutureWarning` (the default becomes `False` in 0.11). |

## The returned dataset — `in_place`

`transform(..., in_place=False)` leaves the dataset untouched and returns an
independent copy: it shares loaders and indices (no data is duplicated) but
owns its pipeline. Transforms already registered are inherited by the copy.
This becomes the default in 0.11.

```python
base = Rellis3DDataset(root, keys=["lidar", "labels"])

v1 = base.transform(augment_v1, in_place=False)
v2 = base.transform(augment_v2, in_place=False)   # independent of v1

base[0]   # raw — base has no transforms
```

`transform(..., in_place=True)` registers the step on the dataset itself and
returns it.

Left unset, `in_place` behaves as `True`, as in 0.9, and `transform()` emits a
`FutureWarning`. Code that assigns the result, `ds = ds.transform(...)`, gives
the same dataset under both values; pass `in_place` to silence the warning.

!!! warning "Until 0.11, transforms register in place by default"
    `v1 = ds.transform(a)` then `v2 = ds.transform(b)` leaves
    `v1 is v2 is ds` with *both* transforms stacked. To build independent
    variants from one dataset, pass `in_place=False`.
