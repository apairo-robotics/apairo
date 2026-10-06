# Idea For Apairo

## Bring your own dataset: the key / order contract

Landed for the asynchronous family (see CHANGELOG). apairo reads a dataset as a
set of channels governed by three per-channel contracts — **order** (list +
order the frames), **load** (decode one frame), **key** (each frame's alignment
value). `load` and `order` were always there; `key` used to be hardwired to a
`timestamps.txt` on disk. It is now pluggable and opt-in in `channels.yaml`:

- `key: {name: <regex>}` — parse the key from the filename stem (groups combined,
  or an explicit `scale: [...]` unit combine); `key: {file: <name>}` — read it
  from a named sidecar. Computed in memory at read time, nothing written.
- `order: {name: <regex>}` — enumeration policy, a contract separate from `key`;
  defaults to the key regex when `key: {name}` is set, else the frame-file
  convention. Required when filenames carry a `_` the default convention rejects
  (a Rellis `<epoch>_<ms>` stem).
- Escape hatch: `self._key_providers[channel]` / `self._order_providers[channel]`
  callables (subclass, set before `super().__init__()`, checked before the YAML
  specs).

This unbricks datasets whose clock lives in the filenames — the Rellis-3D camera
(2847 frames @ 10 Hz) and its ~half-rate image-labels (1200, sparse) now load in
two lines of `channels.yaml`, no subclass, no transcode, zero writes.

Remaining, parked:

- **Unify the synchronous family behind the same key (position-as-default).**

  *First step landed* (see CHANGELOG, "synchronous per-frame clock"): synchrony
  and the presence of a clock are now orthogonal. `is_synchronous` is a
  **structural** flag (co-captured frames — `ds[i]` is one sample across all
  channels), decoupled from `timestamps is None`; and a synchronous
  `ProfiledDataset` now speaks the same key/clock contract as the async family
  for its frame clock — the clock's origin is resolved from a `_clock_provider`
  callable, a profile `clock:` (`{dir, name/units}` / `{file}` per-sequence
  sidecar / `{channel: X}`), or an in-band channel `key:`, always aligned to the
  selected frames by `(sequence, row)`. Rellis carries the co-captured camera's
  clock (without loading the camera); SemanticKITTI its `times.txt`.

  *Remaining:* collapse the two class hierarchies. The end state is
  *position-as-default* — a synchronous dataset is simply the case where every
  channel's `key` is its row position, so `ProfiledDataset` / `SynchronousDataset`
  and `AsyncLayoutDataset` become one order/load/key contract instead of two
  families (and `ds.timestamps` becomes the uniform per-channel form). The engine
  already aligns per-channel key arrays regardless of what the key means; this
  collapse is the larger architectural step, and is deferred.

  *Two guards + a first slice, surfaced by a real eval vault (barakuda, 2026-08):*

  - **Positional default needs the equal-count guard.** "Nothing declared →
    alphabetical order, clockless, synchronous" is the right default **only
    when every channel has the same length** — that is what makes position an
    alignment. On unequal counts (barakuda: 46 / 45 / 6 files per directory)
    positional pairing is silently wrong; the load must refuse loudly and name
    the counts, pointing at `key:` / `timestamps.txt` as the fix.

    *Landed for the clocked case* (see CHANGELOG, "a clock must hold one
    timestamp per frame"): a channel whose `timestamps.txt` or borrowed
    `timestamps_from` clock does not match its frame count is refused at
    construction, and `check` reports it -- and a channel with no clock at all
    -- without loading. The positional default itself does not exist yet
    (a clockless async channel is refused); when it lands, the same
    count-naming refusal applies across channels.
  - **First slice: a bare channel directory is a single-channel dataset.**
    *Landed* (see CHANGELOG, "a directory that holds its data files itself is
    a dataset"), for directories with no sub-directory at all; a folder of
    tables gives one channel per table. The clock rules are unchanged: a bare
    channel still needs a `timestamps.txt` or a declared `key`.
    Data files directly inside the pointed directory (no channel subdirs) →
    one channel named after the directory, alphabetical/numeric order, no
    clock needed since there is nothing to align against. Already expressible
    explicitly (validated): `RawDataset(dir, declare=...)` with
    `directory: "."`; the zero-config detection is the step.
  - **Sequence vs channel is altitude, not identity.** The same directory is a
    *channel* seen from its parent and a *dataset* seen from itself; tools
    should accept both entry points rather than force one reading.

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

Two-tier proposal. *Tier 1 has landed* (see CHANGELOG, "a preprocessor with
several inputs runs on an asynchronous dataset"): the runner zips inputs whose
timestamps are identical and refuses inputs on different clocks by name. It
compares the resolved timestamp arrays rather than following `timestamps_from`
chains, so it also groups derived channels that each carry their own, equal,
`timestamps.txt`. Tier 2 is still open.

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

## Containers: HDF5, Zarr and the episode patterns inside them

*Design note for the first format plugin (`apairo_containers`), after 0.9. It
builds on "The format contract" (R5): the container family is a plugin, not a
core family, and starts with asynchronous recordings.*

### Why a family, and not a loader

Robot-learning data is mostly stored in **containers**: one file (or store)
holding a tree of named N-D arrays. HDF5 and Zarr are the same data model --
groups of arrays, sliced lazily, with attributes -- Zarr being the cloud-native
rework of HDF5. Bolting "a file is a sequence" onto `RawDataset` would mix a
second storage convention into the directory layout family (registry, derived
channels, export all assume a sequence directory). A sibling family contains
that instead: everything after loading -- channel contracts, `synchronize()`,
views, transforms, the clock checks -- is reused as is, and `RawDataset` does
not change.

### What varies is not the format, it is the episode pattern

| Pattern | How a sequence is delimited | Seen in |
|---|---|---|
| **file** | one file per episode | ALOHA / ACT (`episode_N.hdf5`), REASSEMBLE, MIT Push |
| **group** | one group per episode in one file | robomimic, MimicGen, LIBERO (`/data/demo_0/...`) |
| **ends** | arrays concatenated over episodes, an index of episode ends | Diffusion Policy and UMI replay buffers (Zarr: `data/<key>` + `meta/episode_ends`) |
| **flags** | concatenated arrays, an episode-boundary flag per row | D4RL (`terminals`, `timeouts`) |

The same pattern appears in both formats, so the declaration names the
pattern, and the format is read off the file:

```yaml
version: 1
container: {episodes: {files: "*.hdf5"}}     # or {groups: "/data/demo_*"}, {ends: /meta/episode_ends}
channels:
  qpos:   {array: /observations/qpos}
  cam:    {array: /observations/images/cam_high}
  action: {array: /action}
```

### Clocks: two regimes, both already in the model

- **Synchronous episodes** (ALOHA, robomimic, Diffusion Policy): every array
  has one row per step and no clock of its own. The row *is* the clock -- the
  position-as-default of the "unify the families" note, here scoped to a
  container, where it is safe because the arrays come from one writer. The
  equal-count guard applies: arrays of different lengths in one episode are
  refused by name, never paired by position.
- **Asynchronous recordings** (REASSEMBLE, DROID raw): each sensor has its own
  timestamps, a sibling array or a column:
  `key: {array: /timestamps/joint_positions}` or `key: {column: 0}`.

### Naming

`array:` for the path inside the container -- Zarr's word, and neutral across
formats (HDF5 calls it a dataset, which would also collide with apairo's own
"dataset"). It is only valid in a container declaration.

### Backends

A small interface -- open a container, list its arrays, read rows, read an
attribute -- with one implementation per format:

- **HDF5** through `h5py` (`apairo[hdf5]`). Handles are not fork-safe: a file
  is opened lazily, once per process, so a `DataLoader` with workers gets its
  own handle in each. NetCDF4 and MATLAB v7.3 `.mat` files are HDF5 underneath
  and come for free.
- **Zarr** through `zarr` (already `apairo[zarr]`), for stores holding groups.
  The existing `zarr` loader stays: there a channel *directory* is one Zarr
  array, in the directory layout.
- **NPZ** is a flat container read by numpy alone -- a cheap third backend if
  a dataset asks for it.

Two backends from the start keep the interface honest: an HDF5-only design
would grow HDF5-shaped corners.

### Out of the family, deliberately

- **Tables** (Parquet / Arrow -- LeRobot v2 and v3 store steps as Parquet rows
  with an `episode_index` column, frames as MP4): a row store, closer to the
  `csv` loader than to a tree of arrays. Reading LeRobot datasets back would
  close the loop with `apairo_huggingface`, which exports to it -- a later item,
  with the video loader.
- **Encoded media** inside a container (REASSEMBLE's MP4 and MP3 byte strings):
  the video loader.
- **TFRecord / RLDS** (Open X-Embodiment) needs TensorFlow, and **WebDataset**
  tar shards are sequential by design; neither offers the random access apairo
  is built on. Those datasets are reachable through their LeRobot conversions.
- **Message logs** (rosbag, MCAP) stay with `apairo_extractor`.

### Writes

A container is read-only, like every dataset apairo reads. Derived channels
(`run_preprocess`) need a place of their own: a sidecar tree beside the
container, one directory per episode. Until that exists, a preprocess on a
container dataset is refused with that explanation.

### Validation data

- REASSEMBLE: one demonstration range-read out of the 59 GB `data.zip` on TUData
  (async, file pattern, per-sensor timestamps).
- Diffusion Policy Push-T replay buffer (Zarr, ends pattern).
- An ALOHA / ACT simulated episode (HDF5, file pattern, synchronous).

## The format contract

*Design note for ROADMAP R5.*

A format is everything the core needs to know about one way of storing a
channel. Today that knowledge is spread across the core, written per loader
name; the contract gathers it in one object a plugin can provide.

| What the core needs | Where it is hard-coded today | Contract |
|---|---|---|
| Is this directory one of mine? | `_detect_loader` (an if-chain over extensions) | `detect(directory)`, with a `priority` |
| Which files are frames? | `_HINT_EXTS`, `_enumerate`'s extension map | `extensions` |
| One file per frame, or one object per channel? | sets of names in `_init_loaders` and `_verify_key_order` | `per_frame` |
| Is each matching file its own channel? (tables) | `frames != "csv"` in `_bare_channel_entries` | `one_channel_per_file` |
| Build the loader from the channel's metadata | branches on `pcd` / `csv` / `npy` in `_init_loaders` | `open(directory, meta, files)` |
| Clock forms the data provides | `column` keys special-cased in config, keys and dataset | `key_forms`, `clock(loader, spec)` |
| Channel fields it accepts | `array_file` for `npy`/`csv`, `fields` for `pcd`/`csv` | `fields` |
| What `status` prints | per-loader branches in `_channel_shape` and friends | `facts(directory, meta)` |
| What `declare` suggests | `_declare_fields_hint`, `_declare_column_key_hint` | `declare_hints(directory, meta)` |

The generic parts stay in the core and apply to every format: the filename
and sidecar key forms for per-frame formats, `timestamps.txt`, the clock
checks, `synchronize()`, the views. A format only answers questions about its
own bytes.

Registration: built-ins register when `apairo.loader` is imported; plugins
through the `apairo.formats` entry point group, loaded on first use; a name
that collides with a built-in is refused. `str_to_loader` stays as a view of
the registry; `KNOWN_LOADERS` is gone (`format_names()`).

A conformance kit (`apairo.testing.check_format`) runs a plugin's format
against a sample directory it writes: detection, frame count, shape agreeing
with `facts`, a load through `RawDataset` with a `timestamps.txt`, and
validation of its own metadata.

*Landed (R5).* Two decisions made while building it:

- **The core resolves frame files, the format reads them.** A `key`/`order`
  regex and a suffixed variant select files the same way for every per-frame
  format (`apairo.core.naming.channel_frame_files`), and `open()` and
  `facts()` receive the result. Loading and `status` share that one function,
  so they count the same frames.
- **`validate()` checks the fields present, never requires one.** `check`
  reads `channels.yaml` and the declaration apart, and a declaration may
  supply a field the registry entry lacks; a field still missing at load time
  is `open()`'s to refuse.

Out of the contract, on purpose: the write side (`WRITERS`, preprocess
outputs) and the profiled family's readers, which have their own registries.
