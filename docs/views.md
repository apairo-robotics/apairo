# Views & Caching

Two lightweight primitives for controlling **what** gets loaded and **where** it lives in memory.  Both return full apairo datasets — they chain with `.transform()`, `.filter()`, `.join()`, and plug directly into PyTorch `DataLoader`.

---

## `ds.select(keys)` — channel projection

Returns a `ChannelView`: a zero-copy view over a subset of channels.  The parent's transforms are applied first; then only the requested keys are kept.

```python
ds = Rellis3DDataset(root, keys=["lidar", "trav_gt", "ground_height_csf"])
ds.transform("ground_height_csf", expensive_smooth)

view = ds.select(["ground_height_csf"])
view[0].data  # {"ground_height_csf": ...}  — smooth already applied
```

`select()` is most useful as the setup step before `.cache()`.

---

## `ds.cache()` — in-memory materialisation

Returns a `CachedDataset`: iterates the full dataset **once** at call time, stores every sample in RAM, and serves all subsequent accesses from memory with no I/O.

```python
ds_prior = ds.select(["ground_height_csf"]).cache()
# ds_prior is now in RAM — reading it costs no disk I/O
```

> **Memory warning** — all samples are loaded at construction.  Only call `.cache()` on datasets that fit in RAM, typically after `.filter()` or `.select()` has reduced the volume.

---

## The canonical pattern

Cache an expensive derived channel once, reuse it across multiple training configurations:

```python
ds = Rellis3DDataset(root, keys=["lidar", "trav_gt", "ground_height_csf"])
ds.transform("ground_height_csf", expensive_smooth)

# Build the cache once — smooth is computed here and stored in RAM
ds_prior = ds.select(["ground_height_csf"]).cache()

# Each training run reads base channels from disk + prior from RAM
ds_v1 = Rellis3DDataset(root, keys=["lidar", "trav_gt"]).join(ds_prior).transform(augment_v1)
ds_v2 = Rellis3DDataset(root, keys=["lidar", "trav_gt"]).join(ds_prior).transform(augment_v2)

loader_v1 = DataLoader(ds_v1, batch_size=8, shuffle=True)
loader_v2 = DataLoader(ds_v2, batch_size=8, shuffle=True)
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
