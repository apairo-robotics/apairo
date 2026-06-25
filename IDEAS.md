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
so a fast channel under a slow clock is decimated — `"latest"` keeps 1 IMU
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
