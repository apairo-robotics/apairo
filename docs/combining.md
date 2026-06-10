# Combining Datasets

## ZipDataset — merging channels

`ZipDataset` merges channels from multiple datasets of the **same length**. Where `ConcatDataset` concatenates along the frame axis, `ZipDataset` merges along the channel axis: `zip_ds[i].data` is the union of each parent's `data` at index `i`.

```python
ds_base  = Rellis3DDataset(root, keys=["lidar", "trav_gt"])
ds_prior = Rellis3DDataset(root, keys=["ground_height_csf"])

combined = apairo.ZipDataset(ds_base, ds_prior)
sample = combined[0]
# sample.data == {"lidar": ..., "trav_gt": ..., "ground_height_csf": ...}
```

Transforms registered on each parent are applied before merging. The result is a full apairo dataset — `.transform()`, `.filter()`, `.cache()` all chain naturally.

### Fluent form — `ds.join()`

```python
combined = ds_base.join(ds_prior)
```

### Key collisions

By default `ZipDataset` raises at construction if two parents share a key:

```python
ZipDataset(ds_a, ds_b)                          # raises KeyError if keys overlap
ZipDataset(ds_a, ds_b, on_collision="last")     # last dataset wins silently
```

### Three or more datasets

```python
combined = apairo.ZipDataset(ds_base, ds_prior, ds_extra)
# or
combined = ds_base.join(ds_prior, ds_extra)
```

---

## ConcatDataset

`ConcatDataset` concatenates multiple dataset instances into one. It intersects the available keys across all datasets, so every index always returns the same set of modalities.

A single dataset instance already loads all sequences under its root, so `ConcatDataset` is for combining **multiple independent roots** (different drives, different recording sessions):

```python
import apairo

ds1 = apairo.SemanticKittiDataset("/data/kitti_drive1", keys=["lidar", "labels"])
ds2 = apairo.SemanticKittiDataset("/data/kitti_drive2", keys=["lidar", "labels"])

combined = apairo.ConcatDataset([ds1, ds2])
print(len(combined))        # sum of all frame counts
sample = combined[0]        # first frame from the first root
```

Indexing is O(log n) via binary search over cumulative lengths.

---

## Sequence-level splits

`split_sequences` partitions a list of dataset objects at the **sequence level**, avoiding temporal leakage between train and validation sets. It is useful when you have multiple independent roots that you want to split by ratio:

```python
roots = [f"/data/kitti/drive_{i:02d}" for i in range(10)]
datasets = [apairo.SemanticKittiDataset(r, keys=["lidar", "labels"]) for r in roots]

train, val, test = apairo.split_sequences(datasets, ratios=(0.8, 0.1, 0.1))

train_ds = apairo.ConcatDataset(train)
val_ds   = apairo.ConcatDataset(val)
test_ds  = apairo.ConcatDataset(test)
```

For datasets that have a built-in split layout (GOOSE, Rellis-3D), use the `split=` parameter instead:

```python
train_ds = apairo.Goose3DDataset("/data/GOOSE_3D", keys=["lidar", "labels"], split="train")
val_ds   = apairo.Goose3DDataset("/data/GOOSE_3D", keys=["lidar", "labels"], split="val")
```

The split is positional (first 80% of sequences -> train, etc.) rather than random, which preserves temporal structure within each split.

!!! warning "Use sequence-level splits, not frame-level"
    Frame-level random splits on time-series data leak future context into validation. Always split at the sequence boundary.

---

## PyTorch DataLoader

### Synchronous datasets

`ConcatDataset` implements `__getitem__` and `__len__`, which is all PyTorch's `DataLoader` needs for map-style loading. No wrapper class required:

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    apairo.ConcatDataset(train),
    batch_size=8,
    shuffle=True,
    num_workers=4,
)

for batch in train_loader:
    lidar  = batch["lidar"]    # (8, N, 4)
    labels = batch["labels"]   # (8, N)
```

!!! note "Batching point clouds"
    Point clouds have variable numbers of points per frame. Use a custom `collate_fn` to handle variable-length tensors, or pre-process scans to a fixed size in your preprocessor.

### Asynchronous datasets

`DataLoader` checks `isinstance(dataset, IterableDataset)` to decide between map-style and iterable-style loading. For async datasets, inherit from `IterableDataset` to avoid `DataLoader` attempting random access:

```python
from torch.utils.data import IterableDataset, DataLoader

class MyTartanDataset(apairo.TartanKittiDataset, IterableDataset):
    pass

ds = MyTartanDataset("/data/tartan/seq", keys=["velodyne_0"])
loader = DataLoader(ds, batch_size=1, num_workers=0)
```

The one-line subclass is intentional -- you own this integration point and can add custom `collate_fn`, worker init, or sampling logic without touching the library.

---

## Mixing datasets

You can concatenate datasets of different types as long as they share at least one key. The intersection is taken automatically:

```python
kitti = apairo.SemanticKittiDataset("/data/kitti", keys=["lidar", "labels"])
goose = apairo.Goose3DDataset("/data/GOOSE_3D", keys=["lidar", "labels"], split="train")

combined = apairo.ConcatDataset([kitti, goose])
# keys = {"lidar", "labels"}  -- intersection, both share these
```
