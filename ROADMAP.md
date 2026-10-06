# Roadmap

The committed plan for apairo up to the 0.9.0 release and the JOSS submission.
It is fixed: items change only by an explicit edit to this file, dated in the
log at the bottom.

- **`ROADMAP.md`** (this file): what will be done, in what order, and when an
  item counts as done.
- **[`IDEAS.md`](IDEAS.md)**: the design notebook. Open questions, parked
  designs, the reasoning behind an item.
- **[`CHANGELOG.md`](CHANGELOG.md)**: what actually landed.

Every item states its goal, why it matters, what is in and out of scope, when
it is done, where it lives, its size and what it depends on. Sizes: **S** is
about a day, **M** a few days, **L** a week or more.

---

## At a glance

| # | Item | Size | Target week | Status |
|---|---|---|---|---|
| — | `csv` loader, `key: {column}` | M | — | ✅ landed (`20f843e`), unreleased |
| — | A clock must hold one timestamp per frame | S | — | ✅ landed (`b6b6eb0`), unreleased |
| — | `check` reports what loading refuses | S | — | ✅ landed (`53300f9`), unreleased |
| R1 | Dataset guides: TUM RGB-D and EuRoC | S | W40 (28 Sep) | ✅ landed, unreleased |
| R2 | Multi-channel preprocess on asynchronous datasets (tier 1) | M | W41 (5 Oct) | ✅ landed, unreleased |
| R3 | A directory is a dataset | M | W42 (12 Oct) | ✅ landed, unreleased |
| R4 | Schema and status hygiene | S | W42 (12 Oct) | ✅ landed, unreleased |
| R5 | The format contract: a new format is a plugin (branch `feature/format-plugins`) | L | W42–W44 | in progress |
| R6 | Manipulation examples: KUKA F/T, then REASSEMBLE | M | W44–W45 (26 Oct) | planned |
| R7 | Persist a `synchronize()` result | M | W45 (2 Nov) | planned |
| R8 | Release 0.9.0 | S | W46 (9 Nov) | planned |
| J | JOSS track (in `apairo_paper`) | — | W47–W49 | planned |

---

## 0.9.0: tables, clocks, and datasets beyond off-road

The theme of the release is to show that the data model is not specific to
off-road LiDAR. It should read indoor RGB-D, a drone's camera and IMU, and a
robot arm's force/torque sensor and joint states, all in place, and align
them with the clock checks that make those alignments trustworthy.

### R1. Dataset guides: TUM RGB-D and EuRoC

- **Goal.** Two guides show a well-known SLAM dataset loaded with a
  declaration only, with no subclass and no conversion.
- **Why.** The `csv` loader was built for exactly these layouts. A guide turns
  a feature into evidence that the model generalises, and it answers the
  reviewer question "is this only for off-road?". TUM's own `associate.py`
  performs what `synchronize(tolerance=...)` does, which makes a direct
  comparison possible.
- **Scope.** A docs page per dataset (layout, the `apairo.yaml`, a
  `synchronize()` call, what each match's offset tells you). A runnable
  example per dataset. Miniature fixtures in `test/assets/` reproducing each
  layout, with a smoke test that loads them. **Out:** shipping any real data.
- **Done when.** Both guides are in the docs navigation; each example runs
  against its fixture in CI; each guide's `apairo.yaml` has been checked
  against one real sequence downloaded by hand (named in the guide).
- **Placement.** Core docs and tests. **Depends on.** Nothing (the `csv`
  loader has landed).

### R2. Multi-channel preprocess on asynchronous datasets (tier 1)

- **Goal.** A `FramePreprocessor` with two or more `input_keys` runs on an
  asynchronous dataset when its inputs share one clock.
- **Why.** Today it crashes. Reproduced on 2026-09-24: `lidar` plus a `trav`
  channel borrowing its clock through `timestamps_from` gives
  `KeyError: 'trav'`, because `run_preprocess` iterates the interleaved event
  timeline, one channel per sample. Every multi-input preprocessor in
  `apairo_preprocess` (`TraversabilityFromTrajectory`, `GroundHeightFromLabels`,
  `ImageMaskFromPointLabels`, …) therefore only runs on the synchronous
  profiled datasets. The recorded rigs are the ones that need them.
- **Scope.** Same-clock grouping: input channels whose clocks resolve to the
  same timestamps (a `timestamps_from` chain, or identical arrays) are
  zipped into one sample. The output channel borrows that clock. The error
  for inputs on *different* clocks names the channels and points at tier 2.
  **Out:** tier 2, a preprocess over a synchronised view (after R7).
- **Done when.** The reproduction above writes its output channel; a
  two-input preprocessor from `apairo_preprocess` runs on a TartanDrive
  sequence; there are tests for same-clock inputs, for different-clock inputs
  (clear refusal) and for overwrite protection.
- **Placement.** Core (`preprocess/runner.py`). The satellite
  preprocessors need no change. **Design.** `IDEAS.md`, "Multi-channel
  preprocess on asynchronous datasets".

### R3. A directory is a dataset

- **Goal.** Pointing apairo at a directory whose data files sit directly
  inside it, with no channel subdirectories, works without configuration.
- **Why.** Two real cases fail today. `apairo init` on the KUKA F/T recording
  (three CSV tables side by side) answers "no channels". A bare directory of
  `.pcd` or `.npy` frames cannot be opened as a dataset of its own.
- **Scope.** One data format directly in the directory gives one channel,
  named after the directory. Several `.csv`/`.txt` tables directly in the
  directory give one channel per table, named after the file stem, each with
  `directory: "."` and `array_file`. `init`, `status` and `check` accept both
  entry points. **Out:** the positional default across channels (see "After
  0.9").
- **Done when.** `apairo init` followed by `apairo status` works on the KUKA
  recording and on a bare frame directory, with tests for both.
- **Placement.** Core. **Design.** `IDEAS.md`, "First slice: a bare channel
  directory is a single-channel dataset".

### R4. Schema and status hygiene

- **Goal.** `check` stays quiet on datasets apairo's own tools wrote, and
  `status` shows every channel's rate and shape.
- **Why.** Every dataset from `apairo_extractor` raises a warning:
  `transform: unknown field 'source'`. The extractor records the source TF
  topic, but the core schema does not know that field. Older sidecars carry
  `has_timestamps`, which the current apairo no longer writes. Warnings that
  are always present teach users to ignore `check`. Separately, `status`
  prints `-` as the rate of a channel keyed by filename, and `?` as the shape
  of `img`/`pcd`/`bin` channels, although both are cheap to compute.
- **Scope.** Add `source` to the v1 transform fields as provenance. Report
  `has_timestamps` once as deprecated, with the command that removes it. In
  `status`, compute the rate and span of a filename or sidecar `key` from the
  names, and the shape from the first frame's header.
- **Done when.** `check` is clean on the two barakuda extractions, apart from
  the one-line deprecation notice; `status` shows rates for the TUM guide
  channels; tests cover each change.
- **Placement.** Core.

### R5. The format contract: a new format is a plugin

*Redefined on 2026-10-06 after a critical review: R5 was going to add a
container family to the core. Design: `IDEAS.md`, "The format contract".*

- **Goal.** Anyone can add a data format to apairo with a `pip install`, without
  touching the core: a format is one entry point, like a CLI command already is.
- **Why.** Today a format is a core change. Loader names are hard-coded 24
  times across three core modules (the CLI, the configuration, the
  asynchronous dataset), and adding `csv` touched six core files. That
  contradicts the design the paper states -- a closed core, everything
  open-ended outside it through small contracts -- and formats are the most
  open-ended thing there is. It is also what makes apairo extendable by others,
  which a tool meant to become a standard needs more than any single format.
- **Scope.** A public `Format` contract gathering what the core now hard-codes
  per loader: detection, the files a frame is read from, the loader that
  decodes one, the clock forms the format provides (a table column), the facts
  `status` prints, the hints `declare` writes, the fields it accepts. An
  `apairo.formats` entry point group, plus `register_format()` in process.
  Every built-in loader (`npy`, `npys`, `bin`, `img`, `zarr`, `pcd`, `csv`)
  moves onto it, so the core no longer names a format. A conformance kit a
  plugin runs in its own tests, and a docs page with a complete example plugin.
  Developed on the `feature/format-plugins` branch. **Out:** the write side
  (`WRITERS`, preprocess outputs) and the profiled family's readers, which have
  their own registries.
- **Done when.** No format name is hard-coded outside `apairo/loader/` (checked
  by a test); the full suite passes unchanged; an example plugin installed
  through its entry point loads, shows in `status`, is checked by `check` and
  hinted by `declare`, and passes the conformance kit.
- **Placement.** Core (`apairo/core/formats.py`), loaders. **Size.** L.
  **Depends on.** Nothing.

### R6. Manipulation examples: KUKA F/T, then REASSEMBLE

- **Goal.** Two worked examples outside navigation.
- **Why.** Robot-arm data is where multi-rate alignment matters most: F/T at
  about 700 Hz, joints at 100 Hz and cameras at 15–30 Hz. It is also the
  clearest demonstration that apairo is not an off-road tool.
- **Scope.** KUKA LBR Med F/T and IMU (Zenodo 10.5281/zenodo.11096791, CC BY
  4.0, 1.5 MB). Three clocks read in place; the IMU's documented 8.4 ms lag
  shown through `time_offsets()`. Then REASSEMBLE, one demo (TU Wien, CC BY
  4.0), read through the container plugin: F/T and joint states aligned onto
  a camera clock with a tolerance.
- **Done when.** The KUKA example runs in CI against its data, downloaded and
  cached, or against a committed extract if the licence allows it (CC BY
  does, with attribution). The REASSEMBLE example is documented with the
  exact demo file it was run on.
- **Placement.** Core docs and examples. **Depends on.** R3 (KUKA layout)
  and the container plugin (REASSEMBLE).

### R7. Persist a `synchronize()` result

- **Goal.** Freeze a synchronisation as an index matrix in `.apairo` and
  reload it as a synchronous dataset, without copying any data.
- **Why.** Today `synchronize()` recomputes every session, and a result
  cannot be shared or reproduced exactly. It is also the base that tier 2 of
  R2 needs.
- **Scope.** `view.persist(name)` and `RawDataset(root).load_view(name)`. The
  stored state is the index matrix, the validity mask,
  `{reference, method, tolerance}` and a fingerprint of the source channels. A
  stale view is detected and refused. **Out:** `export()` of arbitrary views.
- **Done when.** Round-trip tests pass (persist, reload, the same samples);
  changing a source channel makes the reload refuse with a staleness message;
  the `.apairo` schema documents the view registry.
- **Placement.** Core. **Design.** `IDEAS.md`, "Persist a synchronize() result
  as a reloadable synchronous view".

### R8. Release 0.9.0

- **Scope.** CHANGELOG `[0.9.0]` section, version bump, tag, GitHub release,
  then the Zenodo archive and the PyPI upload. Also `CITATION.cff` (version,
  date), the docs deployment, and the organisation site where it mentions the
  loaders.
- **Done when.** `pip install apairo==0.9.0` works; the Zenodo concept DOI
  resolves to 0.9.0; CI is green on the tag.
- **Depends on.** Whatever of R1–R7 has landed. An item not done by W46 moves
  to "After 0.9" rather than delaying the release.

---

## JOSS track (`apairo_paper`)

| # | Item | Target |
|---|---|---|
| J1 | Measure the RELLIS-3D devkit path against two lines of `apairo.yaml` (paper placeholder 2) | W47 |
| J2 | Measure the fraction of epoch time spent loading on a real training run (paper placeholder 1) | W47 |
| J3 | AI disclosure with exact model versions; funder in the acknowledgements; check the Rerun sentence | W47 |
| J4 | Copy `paper.md` and `paper.bib` into `apairo/paper/`; point the paper at 0.9.0 | W48 |
| J5 | Submit, with the public GitLab history (Nov 2024–Jan 2025) named in the form | from 2026-11-30 |

The last date follows JOSS's rule of more than six months of public history
on the submitted repository. `apairo-robotics/apairo` was created on
2026-05-29; its predecessor, `augustin-bresset/apairo`, on 2026-05-27.

---

## After 0.9 (not scheduled)

Ordered by expected value. Each needs its own entry above before it is
scheduled.

- **Multi-channel preprocess, tier 2.** Run over a synchronised view, with the
  sync parameters stored as provenance (depends on R7).
- **Aggregating `synchronize`.** Return every event between two reference
  ticks, for example all F/T samples between two camera frames, with a
  reducer in `apairo_transform`.
- **Container datasets, as the first format plugin** (`apairo_containers`,
  with `h5py` and `zarr`). Design: `IDEAS.md`, "Containers". Asynchronous
  recordings first (REASSEMBLE, DROID raw), where apairo adds what other
  loaders do not; synchronous robot-learning containers (ALOHA, robomimic,
  Diffusion Policy) only on demand -- they are aligned already, and LeRobot
  and their own loaders read them.
- **Video frame loader** (`mp4`), for RH20T, REASSEMBLE's encoded cameras and
  LeRobot's videos.
- **Reading LeRobot datasets** (Parquet rows with an `episode_index`, MP4
  frames): closes the loop with `apairo_huggingface`, which exports to it.
  Needs the video loader.
- **Export of filtered, transformed or synchronised views** through the
  writers.
- **nuScenes-mini guide.** The canonical multi-rate driving dataset; its
  calibration lives in JSON tables.
- **Satellites.** `apairo_extractor` writes `cameras:` from the bags'
  `camera_info` topics.

## 1.0 criteria

The public API and the `.apairo` format are declared stable at 1.0
(CHANGELOG). Before it:

- a decision on unifying the synchronous and asynchronous families
  (position-as-default), since it touches the on-disk format;
- the positional default, if adopted, with the cross-channel equal-count
  refusal;
- every deprecated field and alias removed;
- the public API fully covered in the API reference.

---

## Working rules

- An item lands as one or more focused commits, each with its tests and its
  CHANGELOG entry, with CI green before the push.
- Work is committed when it is done. Commit dates are never altered.
- Once a week, update the status column and add a line to the log below.

## Log

- **2026-09-24**: roadmap written. Already landed and unreleased: the `csv`
  loader, the clock-coverage guard, the `check` reports.
- **2026-10-01**: R1 landed. Both guides are checked against real sequences
  (TUM `freiburg1_xyz`; EuRoC `V1_01_easy` and `MH_05_difficult`), which is
  what surfaced a fix on the way: `status` and `check` now read the layout
  through `--declare`. EuRoC needed one declaration per hall rather than one
  file, because the tracker directory differs and an undeclared table has no
  clock.
- **2026-10-01**: the `status` half of R4 landed early, because both guides
  showed it: rate and span from a declared `key`, shape from the first frame
  of `img` / `bin` / `pcd` channels. Left in R4: the `source` transform field
  and the `has_timestamps` deprecation notice.
- **2026-10-01**: R2 landed. The runner zips inputs with identical timestamps
  and refuses different clocks by name. Checked on 30 real TartanDrive scans
  with `GroundHeightFromLabels`. Tier 2 (a preprocess over a synchronised
  view) stays in "After 0.9", behind R7.
- **2026-10-06**: R3 landed. A directory with no sub-directory is a dataset:
  its frames are one channel and each table is another. `declare` suggests a
  table's clock column. Checked on the real KUKA recording, which now goes
  through `init`, `declare` and `status` with no edit, and on an evaluation
  export of `.pcd` clouds. Found along the way: EuRoC's `cam0/data.csv`, an
  index file, had been detected as a channel since the `csv` loader landed.
- **2026-10-06**: R4 landed with its schema half: `source` is a transform
  field, and `has_timestamps` is one deprecation line per file. `check` is
  clean on the rosbag barakuda extraction and down to that one line on the
  KITTI-style one.
- **2026-10-06**: R5 redefined from an `hdf5` loader to a container family
  (HDF5 and Zarr backends, episode patterns `files`, `groups`, `ends`), after
  looking at where robot-learning data is stored. It lives on the
  `feature/containers` branch until whole. Part two and a LeRobot reader join
  "After 0.9".
- **2026-10-06**: after a critical review, R5 changes again: before any new
  format, the format contract that lets anyone add one without touching the
  core. Containers become the first plugin on it, after 0.9, asynchronous
  recordings first.
