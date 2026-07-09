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

## Camera intrinsics in Calibration

`Calibration` today is extrinsics only: `{parent}_to_{child}` 4×4 rigid
transforms, resolved with `get_tf`. That covers every lidar↔lidar and
lidar↔base question, but the moment a camera enters the picture (project a
scan into an image, carry per-point labels into pixel space) the caller also
needs the **intrinsics** — K, distortion, image size — and apairo has no home
for them. So they live as loose constructor parameters in satellite
preprocessors (`apairo_preprocess.LidarCameraProjection` takes `intrinsics=`,
`image_size=`, `distortion=`), i.e. copy-pasted per script instead of stored
once as dataset ground truth. The data is sitting right there in the source
recordings (`camera_info` topics in the ROS bags TartanDrive and our own rigs
ship) and is exactly as static as the extrinsics we already persist.

Proposed schema — a `cameras:` section in `.apairo/calibration.yaml`, sibling
of `transforms:`:

```yaml
transforms:
  base_to_velodyne_0: {matrix: [...]}
  base_to_camera_left: {matrix: [...]}
cameras:
  camera_left:
    model: pinhole                     # only model in v1
    K: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    distortion: [k1, k2, p1, p2, k3]   # plumb_bob; optional (rectified: omit)
    size: [height, width]
    frame: camera_left                 # tf frame the projection applies in
```

The `frame` field is the join with the transform graph: it names the tf frame
(the *optical* frame, +Z forward, +X right, +Y down) so a consumer can do
`get_tf("velodyne_0", cam.frame)` then apply K — the two halves of a
projection resolved from one file.

Proposed surface, respecting the stated `Calibration` design ("it resolves;
applying the matrix to data is the caller's job"):

- `Calibration.camera(name) -> CameraModel` — a small frozen dataclass
  (`K`, `distortion`, `size`, `frame`, `model`). Core *stores and returns* the
  model; projecting points through it stays satellite policy
  (`apairo_preprocess`), exactly like `ApplyMatrix` for extrinsics.
- `register_camera(root_dir, name, K, size, distortion=None, frame=None)` —
  writer alongside `register_static_transform`, so extractors
  (`apairo_extractor`) persist `camera_info` at extraction time.
- `read_calibration` stays backward compatible: `cameras:` absent → empty;
  the `Calibration` dict facade over `transforms:` is untouched.

Open design questions, by priority:

- **Optical vs body frame.** ROS rigs carry both `camera_left` and
  `camera_left_optical`. Storing which one `frame` means is the schema's job
  (docstring: it must be the optical frame); validating it is not possible —
  fail-loud is not available here, only convention.
- **Distortion models.** v1 pins `pinhole` + plumb_bob. Fisheye/equidistant
  (`Kannala-Brandt`) is additive later via `model:`; do not design for it now.
- **Per-sequence divergence.** The async family merges per-sequence
  calibration tables; camera entries must merge under the same rule (same
  name, different K across sequences → fail loud).

Motivating consumers already written: `apairo_preprocess`'s projection
preprocessors (`LidarCameraProjection`, `PointFeaturesFromImage`,
`ImageMaskFromPointLabels`) would take `camera=ds.calibration.camera("camera_left")`
instead of three loose arrays; the traversability-mask channel of the planned
TartanDrive HF dataset is the first production pipeline through them.

Placement: apairo core (`Calibration`, `read_calibration`, writer) +
`apairo_extractor` (fill from `camera_info`). Additive schema, post-1.0.

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
