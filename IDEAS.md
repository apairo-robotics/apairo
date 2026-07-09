# Idea For Apairo

## Persist a synchronize() result as a reloadable synchronous view

`.synchronize()` recomputes the matching every session, and nothing lets us
freeze the result and reload the aligned data as a synchronous dataset without
recopying anything.

The insight: a synchronization, reduced to its state, is not data — it's an
index matrix. For each reference frame `i`, each async channel points to a source
row `j` (plus a validity mask for frames with no match within tolerance).
`N_ref × N_channels` integers. This is exactly `filter` generalised: `filter`
persists *one* index array (`np.save(view.indices)`, reload without I/O);
synchronize persisted is *one index column per channel*.

So "register a sync view inside an async dataset" ≠ multiplying datasets. It
means writing that mapping into `.apairo` and reloading it as a synchronous
dataset (`RawDataset`-clean ergonomics) without ever copying a point cloud.

Proposed surface:

```python
sync = ds.synchronize(reference="lidar", method="nearest", tolerance=0.05)
sync.persist("lidar_synced")                       # once
ds2 = apairo.RawDataset(root).load_view("lidar_synced")   # behaves synchronous
```

Persisted state = index matrix `N_ref × N_channels` + validity mask +
`{reference, method, tolerance}` + fingerprint of the source channels.

Open design questions, by priority:
- **Provenance / staleness.** Frozen indices go stale if a source channel
  changes. `filter` already has this; here it's sharper because the mapping also
  encodes the method. The view must carry params + a source-channel
  hash/timestamp so apairo can say "stale → recompute". Otherwise we silently
  reintroduce the `trav_traj` bug (length 19398 ≠ 9701).
- **Addressing.** `channels.yaml` registers channels; need a parallel registry
  of named views in `.apairo` (e.g. `views.yaml`).
- **Reload semantics.** Reloaded view presents as synchronous (single index,
  aligned frames) but keeps the reference timestamp accessible.

Placement: apairo core — view + persistence mechanism in `.apairo`, in the
direct lineage of `filter`/`synchronize`. No satellite covers view persistence.

## Aggregating synchronize: N events per tick, not one

Today the matcher is one event per reference tick (`idx.shape == ref_ts.shape`),
so a fast channel under a slow clock is decimated — `"previous"` keeps 1 IMU
sample of 20 between two lidar ticks and drops the rest. The natural extension is
an *aggregating* match mode that returns, per tick, **all** events in the
interval `(t_prev, t_ref]` (optionally capped: at most `n`, or within a window
`w`), so a frame holds `{"imu": [the 20 samples], "lidar": scan}`.

This stays squarely synchronize's job — it's still clock-driven rate
reconciliation, just with cardinality `N` instead of `1`. Two consequences:

- The matcher return goes **ragged** (a list of index arrays, not one `(N,)`
  array) → a second assembly path in `_load`.
- The sample becomes ragged (`data["imu"]` is a variable-length list), which
  violates numpy-in/out. So it **always** needs a downstream reducer to collapse
  `list[ndarray] -> ndarray` (stack / pad / concat). That reducer is a satellite
  policy (`apairo_transform`), not core — same split as `window()`'s reducer.

Not needed for current work — parked here. Distinct from `window()`: this is
async multi-rate accumulation on a clock; `window()` is index-driven
same-sensor neighbourhoods on an already-ordered (often synchronous) dataset.

## Export a dataset subset as a new self-contained root

Nothing materializes a subset (sequences × channels) of a dataset to a new
root. `.select()` / `.filter_sequences()` are lazy in-memory views; `cache()`
goes to RAM; `run_preprocess` writes into the *same* root. So subsetting today
means dropping to the filesystem (rsync with include/exclude on directory
names), which leaves the copied `.apairo` sidecars stale — they reference
channels that were not copied, breaking the "`.apairo` is the source of truth
on disk" invariant (`apairo status` on the copy reports wrong channels).

The need is proven in-repo: `test/assets/extract_mini_datasets.py` is this
feature hand-rolled (subsample real datasets into valid mini fixtures), and
`benchmarks/soak.py` fabricates its tree by hand. Downstream consumers rsync.

The insight: export is the write-side dual of loading, and the write mechanics
already live in the core (`apairo/writer`, `ChannelWriter` writes frames +
timestamps + registers the channel). The actual feature is **sidecar
regeneration over a selection**.

Proposed surface — a terminal chainable operation, so selection stays the
existing API instead of a parallel kwargs vocabulary (kwargs remain the natural
form for the CLI):

```python
RawDataset(root, keys=["lidar", "trav_gt"]).filter_sequences(["s1", "s2"]).export(dest)
```

```bash
apairo export <src> <dest> --keys lidar trav_gt --sequences s1 s2
```

**v1 scope — structural subset, async family only.** Whole sequences × whole
channels, no transforms, no frame filter: a pure file copy (hardlink/reflink
when same filesystem — near-free) plus regenerated sidecars:

- `channels.yaml` per sequence limited to the exported channels (aliases and
  suffix entries preserved);
- `dataset.yaml` with the exported sequence list;
- `calibration.yaml` copied verbatim (channel-independent);
- third-party sidecar files (e.g. the extractor's `metadata.yaml`) are
  **dropped** — they describe the source, not the subset, and copying them
  reproduces the stale-sidecar problem one level up.

**Derived channels are normalized to self-contained.** The exported channel
always gets its own `timestamps.txt` (copied from the `timestamps_from` source
when the channel has none — the same normalization `run_preprocess` applies to
its outputs). `timestamps_from` is kept as pure provenance ("same clock as"),
so a subset never has dangling clock dependencies regardless of the selection.

**Extension (not v1): materializing export of arbitrary views.** A
frame-filtered, transformed or synchronized view exported by *reading samples
and rewriting through the writers* — renumbered stems, fresh `timestamps.txt`,
re-encoded bytes. Powerful (export a cleaned dataset) but a different regime
from the file-copy path. Complementary to "persist a synchronize() result"
above: persist-view is zero-copy inside the same root; export is a new
self-contained root.

Open design questions, by priority:

- **Suffixed sub-channels.** Exporting `velodyne_0_intensity` without
  `velodyne_0` means copying only `*_<suffix>.npy` plus the shared clock into
  the base directory name — feasible; decide fail-loud vs auto-include.
- **Destination collision.** Refuse a non-empty `dest`? `--merge` /
  `overwrite=` semantics mirroring `init`?
- **Provenance block.** An additive `provenance:` entry in the exported
  `dataset.yaml` (source path, selection) would make circulating subsets
  auditable; the tolerant v1 schema allows it. Utility debated — parked, not
  committed.

Validation criterion: the day `extract_mini_datasets.py` rewrites as a single
`export` call, the feature is right.

Placement: apairo core (it touches the canonical layout and the sidecars) plus
the `apairo export` CLI. Additive API and schema — post-1.0, same treatment as
`apairo add`.
