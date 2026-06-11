# Filtering

`filter()` returns a **view** of the dataset restricted to the frames that pass a predicate. Unlike transforms, filtering changes the length of the dataset — it selects which frames are visible, not how they look.

The result is a `FilteredView` — a proper apairo dataset that supports full chaining: `.transform()`, `.filter()`, and direct use as a PyTorch `DataLoader` source.

---

## `dataset.filter()`

Three forms, same method.

### Sample-level form

```python
ds.filter(fn)   # fn: Sample -> bool
```

`fn` receives the full `Sample` (with transforms applied) and returns `True` to keep the frame:

```python
ds.filter(lambda s: s.data["lidar"].shape[0] > 100)
```

### Per-channel form

```python
ds.filter(key, fn)   # fn: value -> bool
```

`fn` receives `sample.data[key]` **before transforms** and returns `True` to keep the frame. Only the specified channel is loaded during the sweep — faster for large datasets:

```python
ds.filter("trav_gt", lambda gt: (gt == 1).sum() >= 50)
```

### Pre-computed indices form

```python
ds.filter(indices)   # indices: np.ndarray | list[int]
```

Pass a previously saved index array directly. No sweep, no I/O cost:

```python
ds.filter(np.load("cache/valid_indices.npy"))
```

---

## Chaining

`filter()` returns a `FilteredView` which is itself an `AbstractDataset`. Transforms registered on the parent are applied first, then any transforms registered on the view:

```python
ds.transform("lidar", Normalize())           # step 1 — on the full dataset

view = ds.filter("trav_gt", lambda gt: ...)  # step 2 — restrict frames
view.transform("lidar", Voxelize())          # step 3 — only on kept frames

train = DataLoader(view, batch_size=4)
```

Filters also chain:

```python
view = (
    ds
    .filter("trav_gt",  lambda gt:  (gt == 1).sum() >= 50)
    .filter(lambda s: s.data["lidar"].shape[0] > 100)
)
```

---

## Persisting and reloading indices

`filter()` with a predicate is **eager**: it sweeps the full dataset once to build the index list. For large datasets, save the result and reload it on subsequent runs to skip the sweep entirely:

```python
# First run — sweep once
view = ds.filter("trav_gt", lambda gt: (gt == 1).sum() >= 50)
np.save("cache/valid_indices.npy", view.indices)

# Subsequent runs — zero sweep
view = ds.filter(np.load("cache/valid_indices.npy"))
```

`view.indices` returns a `np.ndarray` of `int64` global indices. At 10 M frames this is ~80 MB — negligible.

---

## Splitting a filtered dataset by sequence — `frame_sequence_ids`

A common pattern in cross-validation is to filter once across all sequences, then split by sequence membership per fold — without sweeping the dataset again.

`ProfiledDataset` exposes `frame_sequence_ids`: a `np.ndarray` of shape `(len(ds),)` mapping each global frame index to its sequence ID.  Combined with `view.indices`, this makes fold construction a pure numpy operation:

```python
ds_all      = Rellis3DDataset(root, keys=["lidar", "trav_gt"])
ds_filtered = ds_all.filter("trav_gt", lambda gt: (gt == 1).sum() >= 50)

# One lookup — no extra disk sweep
seq_ids = ds_all.frame_sequence_ids[ds_filtered.indices]

# Per fold — pure numpy masking
for train_seqs, val_seqs in folds:
    ds_train = ds_filtered.filter(np.where(np.isin(seq_ids, train_seqs))[0])
    ds_val   = ds_filtered.filter(np.where(np.isin(seq_ids, val_seqs))[0])
```

The disk is read once; fold construction costs only the numpy masking.

---

## Behaviour summary

| Property | Detail |
|---|---|
| **Eager** | Predicate forms sweep the dataset once at `filter()` call time. |
| **`__len__`** | Returns the number of frames that passed the filter. |
| **Parent transforms** | Transforms registered on the parent are applied before the view's own transforms. |
| **Chaining** | `FilteredView` is a full `AbstractDataset` — `.transform()` and `.filter()` work on it. |
| **`view.indices`** | `np.ndarray` of global indices — saveable and reloadable. |
| **Per-channel sweep** | `filter(key, fn)` loads only the specified channel during the sweep, skipping all other I/O. Synchronous datasets only — on an async dataset, call `synchronize()` first. |
| **`frame_sequence_ids`** | `ProfiledDataset` property — maps each global frame index to its sequence ID, enabling sequence-aware splits after filtering. |
