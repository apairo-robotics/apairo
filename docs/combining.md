# Combining Datasets

## ConcatDataset

`ConcatDataset` concatenates multiple dataset instances into one. It intersects the available keys across all datasets, so every index always returns the same set of modalities.

```python
import apairo

sequences = [
    apairo.SemanticKittiDataset(f"/data/kitti/seq_{i:02d}", keys=["lidar", "labels"])
    for i in range(11)
]

combined = apairo.ConcatDataset(sequences)
print(len(combined))        # sum of all sequence lengths
sample = combined[0]        # first frame from the first sequence
sample = combined[-1]       # not supported — use len(combined) - 1
```

Indexing is O(log n) via binary search over cumulative lengths.

---

## Sequence-level splits

`split_sequences` partitions a list of dataset objects at the **sequence level**, avoiding temporal leakage between train and validation sets.

```python
train, val, test = apairo.split_sequences(sequences, ratios=(0.8, 0.1, 0.1))

train_ds = apairo.ConcatDataset(train)
val_ds   = apairo.ConcatDataset(val)
test_ds  = apairo.ConcatDataset(test)
```

The split is positional (first 80% of sequences → train, etc.) rather than random, which preserves temporal structure within each split.

!!! warning "Use sequence-level splits, not frame-level"
    Frame-level random splits on time-series data leak future context into validation. Always split at the sequence boundary.

---

## PyTorch DataLoader

### Map-style (synchronous datasets)

`TorchConcatDataset` wraps `ConcatDataset` with PyTorch's `Dataset` interface, enabling shuffled sampling and multi-process loading.

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    apairo.TorchConcatDataset(train),
    batch_size=8,
    shuffle=True,
    num_workers=4,
)

for batch in train_loader:
    # batch is a dict[str, torch.Tensor] with an extra batch dimension
    lidar  = batch["lidar"]    # (8, N, 4)
    labels = batch["labels"]   # (8, N)
```

!!! note "Batching point clouds"
    Point clouds have variable numbers of points per frame. Use `collate_fn` to handle variable-length tensors, or pre-pad/sub-sample in your preprocessor.

### Iterable-style (asynchronous datasets)

For asynchronous datasets, use `TorchKittiIterDataset` (iterable-style) or `TorchKittiDataset` (map-style with pre-built index):

```python
from torch.utils.data import DataLoader

iter_ds = apairo.TorchKittiIterDataset(
    apairo.TartanKittiDataset("/data/tartan/seq_001")
)

loader = DataLoader(iter_ds, batch_size=1, num_workers=0)
```

---

## Mixing datasets

You can concatenate datasets of different types as long as they share at least one key. The intersection is taken automatically:

```python
kitti_seqs = [apairo.SemanticKittiDataset(p, keys=["lidar", "labels"]) for p in kitti_paths]
goose_seqs = [apairo.Goose3DDataset(p, keys=["lidar", "labels"]) for p in goose_paths]

combined = apairo.ConcatDataset(kitti_seqs + goose_seqs)
# keys = {"lidar", "labels"}  — intersection, both share these
```
