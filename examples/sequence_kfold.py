"""Sequence-level k-fold cross-validation on RELLIS-3D.

Splitting at the *frame* level leaks temporal correlations (consecutive scans
are nearly identical). This example splits at the *sequence* level so no
sequence appears in both train and val.

Pipeline:
  1. Filter frames with too few traversable points (noisy / edge cases).
  2. Persist the filtered indices so later runs skip the sweep entirely.
  3. Partition the remaining sequences into k folds.
  4. For each fold build train / val views with filter_sequences().
"""

from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from apairo import Rellis3DDataset

ROOT       = Path("/data/RELLIS")
INDEX_DIR  = Path("./kfold_indices")
K          = 5
MIN_POS    = 200   # minimum traversable points to keep a frame

INDEX_DIR.mkdir(exist_ok=True)
filtered_idx_path = INDEX_DIR / "filtered_indices.npy"

ds_full = Rellis3DDataset(ROOT, keys=["lidar", "trav_gt"])

# ---------------------------------------------------------------------------
# Quality filter — sweep runs once, indices persisted for all subsequent runs
# ---------------------------------------------------------------------------

if filtered_idx_path.exists():
    ds_filtered = ds_full.filter(np.load(filtered_idx_path))
else:
    # Per-channel form: only trav_gt is loaded during the sweep, no lidar I/O.
    ds_filtered = ds_full.filter("trav_gt", lambda gt: int((gt == 1).sum()) >= MIN_POS)
    np.save(filtered_idx_path, ds_filtered.indices)

print(f"Frames after quality filter: {len(ds_filtered)} / {len(ds_full)}")

# ---------------------------------------------------------------------------
# K-fold split at the sequence level
# ---------------------------------------------------------------------------

seq_ids = np.unique(ds_filtered.frame_sequence_ids)
print(f"Sequences remaining: {len(seq_ids)}")

rng   = np.random.default_rng(seed=42)
order = rng.permutation(len(seq_ids))
folds = np.array_split(order, K)

for fold_idx, val_order in enumerate(folds):
    train_order = np.concatenate([folds[j] for j in range(K) if j != fold_idx])

    # filter_sequences() maps back onto the already-filtered view — no new sweep.
    ds_train = ds_filtered.filter_sequences(seq_ids[train_order].tolist())
    ds_val   = ds_filtered.filter_sequences(seq_ids[val_order].tolist())

    print(
        f"Fold {fold_idx}: "
        f"train {len(ds_train)} frames ({len(train_order)} seqs) | "
        f"val {len(ds_val)} frames ({len(val_order)} seqs)"
    )

    train_loader = DataLoader(ds_train, batch_size=4, shuffle=True)
    val_loader   = DataLoader(ds_val,   batch_size=4, shuffle=False)

    # ... model.fit(train_loader, val_loader) ...
