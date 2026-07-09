# Transforms

Transforms let you apply callables to channel data **at access time**, without persisting anything to disk. This is the right tool for normalisations, type conversions, augmentations, or any operation cheap enough to run on the fly.

> **Companion library:** [`apairo_transform`](https://github.com/apairo/apairo_transform) ships a collection of ready-made transforms (range filters, normalisation, voxelisation, …) that plug directly into this API.

---

## `dataset.transform()`

A single method, two forms. All steps are registered in order and run as a unified pipeline at access time.

### Per-channel form

```python
ds.transform(key, fn, output=None, keep=True)
```

`fn` receives `sample.data[key]` and returns the transformed value. By default the result overwrites `key` in-place:

```python
ds.transform("lidar", lambda pts: pts[pts[:, 2] > -2])
  .transform("lidar", lambda pts: pts / pts.max())
```

### Sample-level form

```python
ds.transform(fn)   # fn: Sample -> Sample
```

`fn` receives the full `Sample`. Use this when an operation must touch several channels consistently:

```python
def range_filter(sample):
    mask = sample.data["lidar"][:, :3].max(axis=1) < 50.0
    sample.data["lidar"]  = sample.data["lidar"][mask]
    sample.data["labels"] = sample.data["labels"][mask]
    return sample

ds.transform(range_filter)
```

### Preprocessor form

```python
ds.transform(preprocessor)   # a FramePreprocessor instance
```

A `FramePreprocessor` is a callable on a `Sample`, so it plugs into the
pipeline directly: its result is published under its declared `output_key`
(override with `output=`) at access time, nothing is written to disk. This is
the lazy preview of a preprocess — see
[Preprocessing](preprocessing.md#preview-before-materializing). A
`SequencePreprocessor` is rejected (it needs the full sequence at once).

All forms return `self` and compose **in registration order**:

```python
ds.transform("lidar", Normalize())   # step 1
  .transform(range_filter)           # step 2 — sees normalised lidar
  .transform("lidar", Voxelize())    # step 3
```

---

## Publishing a channel — `output`

Pass `output` to write the result of a per-channel transform to a **new key** while leaving the source intact. The published channel is then visible to all subsequent pipeline steps:

```python
ds.transform("lidar", RangeFilter(max=50.0), output="lidar_f")

ds.transform("lidar_f", Normalize())   # branch 1 — reads published channel
ds.transform("lidar_f", Voxelize())    # branch 2 — same source, different op
```

Both branches read from `lidar_f` as it was when it was published, regardless of what the other branch does to it.

---

## Temporary channels — `keep=False`

Set `keep=False` alongside `output` to drop the published channel from the final sample. Useful for intermediate results that are only needed within the pipeline:

```python
ds.transform("lidar", compute_mask_fn, output="_mask", keep=False)
ds.transform(lambda s: apply_mask(s, "_mask"))
# "_mask" is gone from the returned sample; "lidar" and "labels" are filtered
```

---

## `Compose`

`Compose` wraps multiple callables into one, useful for naming or reusing a pipeline:

```python
from apairo import Compose

ds.transform("lidar", Compose([RangeFilter(max=50.0), Normalize()]))
print(ds._pipeline[-1])  # Compose([RangeFilter, Normalize])
```

---

## Behaviour summary

| Property | Detail |
|---|---|
| **No disk writes** | Transforms run in memory at `__getitem__` time. |
| **Order** | All steps (per-channel and sample-level) run in registration order. |
| **Scope** | Per-instance. Transforms on `ds` do not affect another instance at the same path. |
| **`output`** | Publishes result as a new channel; source channel unchanged. |
| **`keep=False`** | Removes an `output` channel from the final sample after the full pipeline runs. |
| **`in_place=False`** | Registers the transform on an independent branch instead of `self`. |

!!! warning "Transforms register in place by default"
    `transform()` mutates the dataset and returns the **same object** for
    chaining. `v1 = ds.transform(a)` then `v2 = ds.transform(b)` leaves
    `v1 is v2 is ds` with *both* transforms stacked. To build independent
    variants from one dataset, pass `in_place=False`.

## Branching — `in_place=False`

`transform(..., in_place=False)` leaves the dataset untouched and returns an
independent branch: a lightweight copy that shares loaders and indices (no
data is duplicated) but owns its pipeline. Transforms already registered are
inherited by the branch.

```python
base = Rellis3DDataset(root, keys=["lidar", "labels"])

v1 = base.transform(augment_v1, in_place=False)
v2 = base.transform(augment_v2, in_place=False)   # independent of v1

base[0]   # raw — base has no transforms
```

Use the default (in place) to build one pipeline step by step; use
`in_place=False` at the point where pipelines diverge.
