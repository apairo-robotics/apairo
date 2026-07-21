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

## Materializing export of arbitrary views

`export()` v1 ships the *structural* subset — whole sequences × whole channels
of the async `RawDataset` family, a pure file copy (`--link` hardlinks on the
same filesystem) with the `.apairo` sidecars regenerated so the copy is
self-contained (`apairo status` on it reports exactly the exported channels).
The remaining regime is exporting a **frame-filtered, transformed or
synchronized** view, by *reading samples and rewriting through the writers* —
renumbered stems, fresh `timestamps.txt`, re-encoded bytes. Powerful (export a
cleaned or resampled dataset to a new root) but a different mechanism from the
zero-read file copy; complementary to "persist a synchronize() result" above
(persist-view is zero-copy inside the same root; export is a new self-contained
root).

The v1 guard rejects such views with a pointer here, so the call site fails loud
rather than silently copying the pre-filter data. The textbook case is
`build_gt3d_pack` in apairo_experiments: select the real-GT frames, voxelize,
and renumber into a fresh (97 GB) root — frame-filter + transform + renumber,
exactly the v1-excluded path.

Open, still deferred:

- **Provenance block.** An additive `provenance:` entry in the exported
  `dataset.yaml` (source path, selection) would make circulating subsets
  auditable; the tolerant schema allows it. Utility debated — parked.

Validation criterion: the day `test/assets/extract_mini_datasets.py` (which
frame-windows *and* point-subsamples, and handles synchronous Rellis) rewrites
as a single `export` call, the extension is right.

Placement: apairo core + the `apairo export` CLI — both already carry v1.

## Multi-channel preprocess on asynchronous datasets

`run_preprocess` builds `dataset_cls(root, keys=preprocessor.input_keys)` and
iterates. On the async family that iteration is the **interleaved event
timeline** — one key per sample — so any preprocessor with two or more input
keys crashes (`KeyError`) the moment it runs on a raw dataset: the sample
never holds both channels. Every multi-input preprocessor in
`apairo_preprocess` (`TraversabilityFromTrajectory`, `GroundHeightFromLabels`,
`TrajectoryDistance`, `ImageMaskFromPointLabels`) therefore only runs on the
profiled synchronous datasets (Rellis, GOOSE) — yet the datasets that *have*
cameras and trajectories worth preprocessing (TartanDrive, our own rigs) are
all async. This is the `trav_traj` length bug (19398 ≠ 9701) seen from the
other side: 19398 *is* the two-channel interleaved timeline.

The workaround today is manual and lossy in ergonomics: build a
`synchronize()` view, pull samples, call the preprocessor directly, persist
with `ChannelWriter` — four steps re-implementing what `run_preprocess` does
in one, minus overwrite protection and provenance defaults.

Two-tier proposal:

- **Cheap tier — same-clock grouping.** Channels sharing an identical clock
  (`timestamps_from` chains resolving to the same `timestamps.txt`) are
  *already* aligned; interleaving them as separate events is pure loss. The
  runner (or the async `_load` under a flag) can zip same-clock channels into
  one sample instead. This alone unlocks the derived-channel compositions —
  `trav_traj` + `lidar_uv` are both on the lidar clock by construction.
- **General tier — preprocess over a synchronized view.** Let the runner
  accept sync parameters and build the view itself:

  ```python
  TartanKittiDataset.run_preprocess(
      ImageMaskFromPointLabels(...), root,
      sync={"reference": "velodyne_0", "tolerance": 0.05},
  )
  ```

  Output channel timestamps = the reference clock; `sync` params recorded in
  `.apairo` as provenance. This is the natural consumer of "persist a
  synchronize() result" above — a persisted view makes the sync reproducible
  instead of re-derived per run — but it does not depend on it.

Open design questions, by priority:

- **Provenance.** A channel derived *through* a sync is only reproducible if
  the sync params (reference, method, tolerance) are stored with it;
  otherwise re-running with different params silently changes the channel.
- **SequencePreprocessor.** Same gap, same fix (the sequence runner already
  materializes `frames = [dataset[i] ...]`); lazy `transform()` stays
  frame-only.
- **Tolerance drops.** Frames dropped by `tolerance` leave holes in the
  output clock — fine (the channel gets its own `timestamps.txt`), but the
  `timestamps_from` shortcut no longer applies; the runner must detect this.

Placement: apairo core (`preprocess/runner.py`, possibly a flag on the async
`_load`). The satellite preprocessors need zero changes — that is the point.

## Camera intrinsics in the core calibration

Scheduled pre-1.0 -- core part shipped (see CHANGELOG). `Calibration` used to
hold extrinsics only; lidar->image projection downstream needs `K` +
distortion, and passing them as preprocessor parameters would move static rig
config out of `.apairo` (repaid per dataset, invisible to `apairo status`).

The split follows the `get_tf` precedent verbatim: **storing and exposing**
intrinsics is core (static rig config in `calibration.yaml`, a `cameras:`
section mirroring ROS `CameraInfo` field names); **applying** them
(projection, undistortion) is model-dependent and stays in `apairo_transform`.
Entries are keyed by the camera's *frame* (`CameraInfo.frame_id`) -- one entry
per physical camera; image channels reach it via their `frame` field in
`channels.yaml`.

Remaining, outside this repo:

- **apairo_extractor**: write `cameras:` entries from the rosbags'
  `camera_info` topics (near-verbatim -- the schema mirrors the message).
- **apairo_transform**: the projection/undistortion ops consuming
  `ds.calibration.get_intrinsics(...)` (e.g. `ProjectPoints`), including the
  lidar->image preprocessor that motivated this.
