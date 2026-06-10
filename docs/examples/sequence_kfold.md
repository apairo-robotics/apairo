# Sequence-level k-fold cross-validation

**Scenario:** Evaluate a model with k-fold CV on RELLIS-3D, ensuring no sequence leaks between train and val (consecutive scans are near-identical — frame-level splits inflate metrics).

**APIs:** `filter()`, `frame_sequence_ids`, `filter_sequences()`, persisted indices.

---

## Quality filter — run once, persist

Remove frames with too few traversable points.  The per-channel form of `filter()` loads only `trav_gt` during the sweep, skipping lidar I/O entirely.  Save the result so subsequent runs skip the sweep.

```python
import numpy as np
from apairo import Rellis3DDataset

ROOT      = "/data/RELLIS"
INDEX_DIR = "./kfold_indices"
MIN_POS   = 200

ds = Rellis3DDataset(ROOT, keys=["lidar", "trav_gt"])

filtered_path = f"{INDEX_DIR}/filtered_indices.npy"

if Path(filtered_path).exists():
    ds_filtered = ds.filter(np.load(filtered_path))
else:
    ds_filtered = ds.filter("trav_gt", lambda gt: int((gt == 1).sum()) >= MIN_POS)
    np.save(filtered_path, ds_filtered.indices)
```

## K-fold split at the sequence level

After filtering, ask which sequences are still represented.  Then partition *those sequence IDs* into k folds — not individual frames.

```python
seq_ids = np.unique(ds_filtered.frame_sequence_ids)

rng   = np.random.default_rng(seed=42)
order = rng.permutation(len(seq_ids))
folds = np.array_split(order, k)
```

## Building train / val per fold

`filter_sequences()` maps back onto the already-filtered view.  No new disk sweep — it's a numpy index selection on top of the existing `FilteredView`.

```python
from torch.utils.data import DataLoader

k = 5
for fold_idx, val_order in enumerate(folds):
    train_order = np.concatenate([folds[j] for j in range(k) if j != fold_idx])

    ds_train = ds_filtered.filter_sequences(seq_ids[train_order].tolist())
    ds_val   = ds_filtered.filter_sequences(seq_ids[val_order].tolist())

    train_loader = DataLoader(ds_train, batch_size=4, shuffle=True)
    val_loader   = DataLoader(ds_val,   batch_size=4, shuffle=False)

    # ... model.fit(train_loader, val_loader) ...
```

!!! warning "Sequence-level, not frame-level"
    Random frame-level splits on time-series data create temporal leakage: frames at index `i` and `i+1` are almost identical, so a model trivially generalises across the split.  Always partition at the sequence boundary.
