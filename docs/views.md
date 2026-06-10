# Views & Caching

Two lightweight primitives for controlling **what** gets loaded and **where** it lives in memory.  Both return full apairo datasets — they chain with `.transform()`, `.filter()`, `.join()`, and plug directly into PyTorch `DataLoader`.

---

## `ds.select(keys)` — channel projection

Returns a `ChannelView`: a view over a subset of channels.  The parent's transforms are applied first; then only the requested keys are kept.

```python
ds = Rellis3DDataset(root, keys=["lidar", "trav_gt", "ground_height_csf"])
ds.transform("ground_height_csf", expensive_smooth)

view = ds.select(["ground_height_csf"])
view[0].data  # {"ground_height_csf": ...}  — smooth already applied
```

`select()` is most useful as the step before `.cache()`.

---

## `ds.cache()` — in-memory materialisation

Returns a `CachedDataset`: iterates the full dataset **once** at call time, stores every sample in RAM, and serves all subsequent accesses from memory with no I/O.

```python
ds_prior = ds.select(["ground_height_csf"]).cache()
# ds_prior is now in RAM — reading it costs no disk I/O
```

> **Memory warning** — all samples are loaded at construction.  Only call `.cache()` on datasets that fit in RAM, typically after `.filter()` or `.select()` has reduced the volume.

---

## `.cache()` as a deterministic boundary

The most important property of `.cache()` is not performance — it's **what it communicates**.

The rule is simple:

- **Deterministic** → safe to cache: dtype casts, coordinate transforms, filters, preprocessed channels
- **Stochastic** → never cache: data augmentation, random subsampling, dropout

Placing `.cache()` mid-chain makes the boundary explicit and visible at a glance:

```python
# Everything before .cache() is deterministic — computed once, frozen in RAM
ds_train_base = (
    ds.split("train")
    .filter("trav_gt", HasMinPositives(min_pos))  # deterministic filter
    .transform("lidar", RobotFilter(d=1.0))        # deterministic transform
    .cache()                                        # <-- boundary
)

# Everything after .cache() is stochastic — runs fresh every access
ds_train = ds_train_base.transform(SparseAugment(voxel_size))
```

Without `.cache()`, the boundary exists but is invisible — a reader must trace mentally through the pipeline to find where determinism ends. With `.cache()`, it is structural.

---

## Caching a derived channel across training runs

Cache an expensive derived channel once, reuse it across multiple training configurations:

```python
ds = Rellis3DDataset(root, keys=["lidar", "trav_gt", "ground_height_csf"])
ds.transform("ground_height_csf", expensive_smooth)  # deterministic

# Computed once, stored in RAM
ds_prior = ds.select(["ground_height_csf"]).cache()

# Each training run: prior served from RAM, augmentation applied fresh each access
ds_v1 = Rellis3DDataset(root, keys=["lidar", "trav_gt"]).join(ds_prior).transform(augment_v1)
ds_v2 = Rellis3DDataset(root, keys=["lidar", "trav_gt"]).join(ds_prior).transform(augment_v2)
```

---

## Behaviour summary

| | `select(keys)` | `cache()` |
|---|---|---|
| **When evaluated** | At access time | At construction (eager) |
| **I/O cost** | Same as parent | Zero after construction |
| **RAM cost** | None | All samples in memory |
| **Parent transforms** | Applied before projection | Applied before storing |
| **Chaining** | Full (`transform`, `filter`, `join`, `cache`) | Full |
| **What to put before** | Anything | Deterministic operations only |
| **What to put after** | — | Stochastic operations (augmentation) |
