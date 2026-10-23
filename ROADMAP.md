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
| — | `check` needs no `apairo init` | S | — | ✅ landed (`729baff`), unreleased |
| R1 | Dataset guides: TUM RGB-D and EuRoC | S | W40 (28 Sep) | ✅ landed, unreleased |
| R2 | Multi-channel preprocess on asynchronous datasets (tier 1) | M | W41 (5 Oct) | ✅ landed, unreleased |
| R3 | A directory is a dataset | M | W42 (12 Oct) | ✅ landed, unreleased |
| R4 | Schema and status hygiene | S | W42 (12 Oct) | ✅ landed, unreleased |
| R5 | The format contract: a new format is a plugin | L | W42–W44 | ✅ landed, unreleased |
| R6 | Manipulation example: the KUKA F/T and IMU logs, and `latency` | M | W44–W45 (26 Oct) | ✅ landed, unreleased |
| R7 | Persist a `synchronize()` result | M | W45 (2 Nov) | ✅ landed, unreleased |
| R8 | Release 0.9.0 | S | W46 (9 Nov) | first cut on `release/0.9.0`, to cut again |
| U1 | `ds[-1]`, `print(ds)`, `sample[key]` | S | W43 (19 Oct) | ✅ landed, unreleased |
| U2 | Installation names the ecosystem | S | W43 (19 Oct) | ✅ landed, unreleased |
| U3 | KITTI Odometry through `SemanticKittiDataset` | S | W44 (26 Oct) | planned |
| U4 | "KITTI" names KITTI only; when to use which storage | M | W44–W45 | ✅ landed, unreleased |
| U5 | Satellites on PyPI, then `apairo[extractor]` and the like | M | after 0.9.0 | planned |
| D1 | One docs site for the ecosystem, built from the satellites | M | W47 (16 Nov) | planned |
| D2 | Executed examples for `apairo_transform` | S | W48 (23 Nov) | planned |
| D3 | A screen recording of `apairo studio` | S | W48 (23 Nov) | planned |
| D4 | Studio live on a Hugging Face Space | M | after JOSS | planned |
| D5 | Real-data results for `apairo_preprocess` | M | after JOSS | planned |
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

### R6. Manipulation example: the KUKA F/T and IMU logs, and `latency`

*Redefined on 2026-10-08. The first scope said the IMU's documented 8.4 ms
lag would show "through `time_offsets()`". It cannot: `time_offsets()`
compares timestamps, and a transport delay makes them all late together. A
delay shows only in the data, and is corrected only by a declaration.
REASSEMBLE needs the container plugin, so it moves to "After 0.9".*

- **Goal.** A worked example outside navigation, on a robot arm, and a
  declarative correction for a sensor that stamps its readings late.
- **Why.** Robot-arm data is where multi-rate alignment matters most: F/T at
  about 700 Hz, joints at 100 Hz and cameras at 15–30 Hz. It is also the
  clearest demonstration that apairo is not an off-road tool. A known sensor
  latency is common, and must not be corrected by hand in every script.
- **Scope.** A `latency` channel field, in seconds, that moves the channel's
  clock back before any alignment, whatever the clock's source. The KUKA LBR
  Med F/T and IMU logs (Zenodo 10.5281/zenodo.11096791, CC BY 4.0): a
  shipped declaration (`kuka_ft_imu`) with the documented 8416 µs on the IMU,
  a real excerpt as its sample, a guide, and an example. The example measures
  the delay in the data (IMU acceleration against F/T force, while the arm
  rotates) and checks the declared one against it.
- **Done when.** The example runs in CI on the excerpt. The guide's figures
  come from the full record. The sample passes `check_dataset`.
- **Placement.** Core (`latency`), docs, examples. **Depends on.** R3 (the
  logs are a folder of tables).

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
  loaders. Then, in `apairo_preprocess`, merge `feature/output-format-from-input`:
  it drops the `output_loader = "npys"` declarations and needs `apairo>=0.9`,
  because 0.8 reads that field unconditionally.
- **Done when.** `pip install apairo==0.9.0` works; the Zenodo concept DOI
  resolves to 0.9.0; CI is green on the tag.
- **Depends on.** Whatever of R1–R7 has landed. An item not done by W46 moves
  to "After 0.9" rather than delaying the release.

---

## What a new user hits first (from the tutorial)

The U items come from tutorial sessions in which the author uses apairo as a
newcomer would: a fresh virtualenv, the docs only, real data. The friction
journal lives outside the repositories, in `~/dev/apairo/tutorial/journal.md`.
Session 1 (2026-10-15) installed 0.9.0 from `release/0.9.0`, then loaded the
`barakuda1` rosbag extraction with `RawDataset`.

U1–U4 are small enough to join 0.9.0 if they land before R8 publishes. The
release is already cut on `release/0.9.0`, so including them means cutting it
again on top of them.

### U1. `ds[-1]`, `print(ds)`, `sample[key]`

- **Goal.** The three things a first session tried and that failed now work.
- **Why.** Each is a Python habit: `ds[-1]` raised `IndexError: Index -1 out
  of range [0, 21236)`, `print(ds)` showed `<RawDataset object at 0x…>`
  while every view already had a summary, and `sample["timestamp"]` raised
  `'Sample' object is not subscriptable`.
- **Scope.** Negative integer indices in `AbstractDataset.__getitem__` and
  `SequenceView` (the profiled `(sequence, index)` tuple is unchanged). One
  `__repr__` on `AbstractDataset` (length, sequences on a root, keys, root),
  built from what the dataset holds; views keep theirs. `Sample.__getitem__`
  and `__contains__` over `data`; `"timestamp"` stays a channel name, and the
  `KeyError` for a missing one points to `sample.timestamp`. A `Sample` stays
  a dataclass, not a mapping and not iterable, so `torch`'s
  `default_collate` cannot rebuild it without its timestamp. **Out:** slices
  (`ds[a:b]`).
- **Done when.** Tests cover each case; the Quickstart shows them; checked
  on the real `barakuda1` extraction.
- **Placement.** Core. Branch `feature/tutorial-s1-ergonomics`.

### U2. Installation names the ecosystem

- **Goal.** `installation.md` says what each satellite does and how to
  install it today.
- **Why.** The installation page lists the dataset extras but no satellite.
  A newcomer looked for `pip install apairo[extractor]`. The extractor
  appears in the README only, and not at all in the docs.
- **Scope.** An "Ecosystem" section: `apairo_extractor`, `apairo_transform`,
  `apairo_preprocess`, `apairo_visu`, one line each, with the install
  command that works today (from git, none is on PyPI) and the
  `apairo <command>` it adds when it does. `apairo_visu` adds three, each
  behind its own extra: `apairo visu` (`open3d`), `apairo studio` (`studio`)
  and `apairo rerun` (`rerun`). `apairo_rr` is not listed: it was folded
  into `apairo_visu` as its Rerun backend and is deprecated. **Out:** the
  extras (U5).
- **Done when.** Each command was run in a fresh virtualenv.
- **Placement.** Core docs; the README links to the section.

### U3. KITTI Odometry through `SemanticKittiDataset`

- **Goal.** The docs say whether plain KITTI Odometry loads, and how.
- **Why.** A newcomer looks for `apairo[kitti]` and finds only
  `semantic-kitti`. The SemanticKITTI profile reads the KITTI Odometry
  layout (`sequences/XX/velodyne/*.bin`, `times.txt`) and adds `labels/`, so
  `SemanticKittiDataset(root, keys=["lidar"])` should read KITTI Odometry
  unchanged. Nobody has checked it.
- **Scope.** Check on one real KITTI Odometry sequence; say it in
  `installation.md` and `datasets/index.md`; decide whether a `kitti` extra
  is worth an alias. State that KITTI Raw (`<drive>/<sensor>/data/`, one
  `timestamps.txt` per sensor written as dates) is a different layout that
  no class reads today. **Out:** a KITTI Raw declaration (needs a date clock
  parser; a candidate for "After 0.9").
- **Done when.** The sequence checked is named in the docs.
- **Placement.** Core docs; a fixture test if the check needs a code change.

### U4. "KITTI" names KITTI only; when to use which storage

- **Goal.** One name per thing, and the docs say which storage to choose.
- **Why.** "KITTI" names four things today: the extractor's export, the
  "asynchronous (KITTI-layout)" datasets of `concepts.md`,
  `TartanKittiDataset`, and SemanticKITTI, which is synchronous. The
  extractor's export is not KITTI: it writes `.npy` files with epoch-second
  `timestamps.txt` and a `channels.yaml`, the `.apairo` layout, which KITTI
  tools do not read. A user with real KITTI data who trusts the name fails.
  A newcomer also asked whether this layout is meant as a standard, when
  Zarr and HDF5 suit training better. The answer follows from the design: the
  standard is the `.apairo` contract (the sidecar that describes data where
  it is), not one storage; the per-channel directory is the default because
  it needs only numpy and `ls` can read it.
- **Scope.**
  - `apairo_extractor`: name the export after its storage (for example
    `files` and `zarr`, both with the `.apairo` sidecar), in the TUI, the
    README and the package description ("Extract ROS bags to KITTI format").
  - `concepts.md`: "the per-channel directory layout" instead of
    "KITTI-layout".
  - A short docs section on storage choice: directories of files for
    inspection and for frames read one at a time; Zarr or HDF5 (through a
    `Convert` preprocess, or the containers plugin after 0.9) for many small
    frames, compression, network filesystems, or a single file to ship.
  - The `RawDataset` name ("raw" also reads as KITTI Raw or "unprocessed")
    is an open question. Record it in `IDEAS.md`; any rename waits for 1.0
    and goes through a deprecated alias.
- **Done when.** `grep -ri kitti` over the docs and the extractor only
  finds the KITTI datasets themselves.
- **Placement.** Core docs and `apairo_extractor`.

### U5. Satellites on PyPI, then `apairo[extractor]` and the like

- **Goal.** `pip install apairo[extractor]` (and `transform`, `preprocess`,
  `visu`) installs the satellite.
- **Why.** It is how a newcomer expects to find the rest of the ecosystem,
  and it matches the dataset extras.
- **Scope.** Publish `apairo-extractor`, `apairo-transform`,
  `apairo-preprocess` and `apairo-visu` on PyPI (all four names were free on
  2026-10-19). `apairo-rr` is deprecated and is not published. Replace
  `apairo_preprocess`'s `git+https` dependency on `apairo_transform`, which
  PyPI refuses. Then add the extras to apairo, each with a compatible version
  range that is updated at every release; whether `apairo[visu]` also brings
  a backend, such as `apairo-visu[rerun]`, is decided then. Whether
  `apairo_transform` and `apairo_preprocess` publish as one package, with
  two extras, is decided before the names are taken. An extra is an
  install shortcut only: apairo still never imports a satellite.
- **Done when.** Each extra installs in a fresh virtualenv and its
  `apairo <command>` runs.
- **Depends on.** R8.


---

## Documentation: one site for the ecosystem

The core docs are the only docs site. Each satellite gets a page in it, while
its sources (docstrings, examples) stay in its own repository and are pulled
in at build time, so the docs show the code that runs. Satellite READMEs
shrink to the install line and a link. Decided on 2026-10-23: one site
(one search, working cross-links, one build to maintain, one site for the
JOSS reviewers) rather than a docs site per repository.

D1–D3 follow 0.9.0; D4 and D5 follow the JOSS submission.

### D1. One docs site for the ecosystem, built from the satellites

- **Goal.** An "Ecosystem" tab with one page per satellite: what it does,
  its install line, its `apairo <command>`, its examples and its API.
- **Why.** U2 names the satellites on the installation page, but their
  documentation lives in four READMEs, outside the site and its search.
- **Scope.** `docs.yml` checks out `apairo_extractor`, `apairo_transform`,
  `apairo_preprocess` and `apairo_visu` beside the core and installs them.
  `mkdocstrings` documents their public API; `pymdownx.snippets` includes
  their `examples/*.py` as they are. A satellite change reaches the site at
  the next core build: a weekly scheduled build, or a `repository_dispatch`
  from each satellite. The satellite READMEs keep the install line and link
  to their page. `apairo_visu/docs/` and `apairo_huggingface/docs/` move into
  the site or are dropped. **Out:** executing examples (D2), recordings (D3).
- **Done when.** `mkdocs build --strict` passes in CI with the four
  satellites checked out; each page's examples are included from its
  repository, not copied.
- **Placement.** Core docs and `docs.yml`; the satellite READMEs.
- **Depends on.** U2, R8.

### D2. Executed examples for `apairo_transform`

- **Goal.** The transform page shows each example's output, produced by the
  docs build.
- **Why.** A transform is easier to judge by its output (a cloud cropped, a
  label remapped, an augmentation's jitter) than by its code.
- **Scope.** `markdown-exec` (or `mkdocs-gallery`) runs `augmentation.py`,
  `compose.py`, `labels.py` and `pointcloud.py` at build time on a mini
  dataset from the core's `test/assets` (`mini_semantic_kitti`, 1.8 MB for
  all of them), and renders their text output
  and a matplotlib figure (a bird's-eye view of a cloud). The examples are
  adapted to take the fixture's path. **Out:** real datasets, which CI
  cannot download.
- **Done when.** A wrong example fails the docs build.
- **Placement.** `apairo_transform/examples`, core docs.
- **Depends on.** D1.

### D3. A screen recording of `apairo studio`

- **Goal.** The Studio page shows the studio in use.
- **Why.** Studio is a FastAPI server over live dataset objects; GitHub
  Pages is static and cannot run it. A recording is the cheapest way to show
  it.
- **Scope.** A short recording (GIF or MP4, under 30 s, light and dark) of
  `apairo studio` on a public dataset: the pipeline graph, a node clicked,
  its channel table. A script or a written procedure to record it again when
  the UI changes.
- **Done when.** The recording is on the Studio page and its procedure is
  committed.
- **Placement.** Core docs (the Studio page from D1).
- **Depends on.** D1.

### D4. Studio live on a Hugging Face Space

- **Goal.** A reader can use the studio from the docs, without installing
  anything.
- **Scope.** A Docker Space running `apairo studio` on a small public
  dataset, embedded in the Studio page by iframe, as was done for toaster
  (its `hf-space` branch). The free tier sleeps when idle; the page says so.
  A static, serverless demo mode (graph and samples frozen to JSON, served
  by the zero-build front) is the alternative if the Space proves too slow.
- **Done when.** The Space starts from the page and serves the graph.
- **Depends on.** D3, the JOSS submission.

### D5. Real-data results for `apairo_preprocess`

- **Goal.** Each preprocessor's page shows what it produces on a real
  recording.
- **Why.** Ground segmentation, KISS-ICP odometry and traversability are
  judged on real data, which CI cannot download.
- **Scope.** A script in `apairo_preprocess` renders each example on a named
  public sequence into images or short videos, committed with the dataset
  and the command that produced them. 3D results also as a Rerun `.rrd`
  served with the docs and opened in Rerun's web viewer, so the cloud can be
  turned. **Out:** running them in CI.
- **Done when.** Every image says which sequence and which command produced
  it, and the script regenerates them all.
- **Placement.** `apairo_preprocess`, core docs.
- **Depends on.** D1, the JOSS submission.

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
  recordings first -- REASSEMBLE (one demo: F/T and joint states aligned onto
  a camera clock, moved here from R6), DROID raw -- where apairo adds what other
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
- **2026-10-06**: R5 landed on `feature/format-plugins`. The seven built-in
  formats implement `Format`; the CLI, the configuration checks and the
  asynchronous dataset ask the registry, and a test keeps format names out of
  the read path. The `.xyz` example plugin goes through `init`, `declare`,
  `check`, `status`, loading and `synchronize()` from its entry point, and
  passes `check_format` like every built-in. The core shrank by about 200
  lines. Compared with `main` on seven real datasets, the only difference is
  a fix: `status` now reads a cloud export's shape off the frame loading
  calls frame 0. Merge to `main` pending.
- **2026-10-06**: R5 widened before the merge, after review with the user:
  - **Writing joined the contract.** A preprocess writes in the run's
    `output_format`, else the preprocessor's `output_loader`, else its
    input's format when it can hold the output, else `npys` / `npy`. A
    conversion is a preprocess (`Convert`). Profiled datasets read their
    derived channels through the contract.
  - **A table is counted, not parsed.** `status` on a 600,000-row log went
    from 2.95 s to 0.63 s.
  - **Standard datasets install by name.** `pip install apairo[tartan]` and
    the others, the TUM and EuRoC declarations by name, `--as` from a registry
    open to packages.
  - `apairo_preprocess`'s suite passes against the branch.
- **2026-10-06**: the standard datasets audited, at the user's request. Only
  Rellis and TartanDrive had real samples. GOOSE and SemanticKITTI were tested
  on trees their own tests invented.
  - A real GOOSE excerpt found its labels unmasked (now fixed) and its
    clock unread (now read).
  - `check_dataset`, the sample cards with a CI completeness rule, and
    `pytest -m realdata` landed.
  - On the local full copies, Tartan passes. Rellis passes except its clock,
    because the local copy has no camera directory.
  - SemanticKITTI's clouds still wait for a KITTI registration (the user's
    call).
- **2026-10-08**: R6 redefined and landed on `feature/kuka-latency`.
  - **A redefinition.** The plan to show the KUKA IMU's 8416 µs delay
    through `time_offsets()` was wrong: timestamps cannot show a delay they
    all share. It is measured in the data instead. On the full record, the
    IMU's acceleration and the F/T force line up best with the IMU +7.6 ms
    and +7.2 ms late on the first two runs. On the contact run the hand's
    force swamps gravity, and no delay can be measured.
  - **`latency` channel field.** Applied to every clock source. The KUKA
    declaration declares the documented delay with it. After the correction,
    the residual is −0.8 and −1.2 ms, within the method's precision.
  - **`check_dataset` validates the external declaration too**, as
    `apairo check --declare` does. It had missed that a first draft of the
    KUKA declaration did not stand on its own.
  - **A gap.** A folder's tables without a clock (the KUKA calibration
    poses) block opening the folder without `keys=`; see `IDEAS.md`.
- **2026-10-08**: R7 landed on `feature/persist-views`.
  - `persist(name)` and `load_view(name)` work per sequence, and roots keep
    one view per sequence. A view is refused as stale on any change to a
    source's frame count or clock hash, a declared `latency` included.
  - Interpolators are passed again at load; custom matchers are not needed.
  - On a real TartanDrive sequence the frames reload identical, from a 162 KB
    index file. The matching takes about 1 ms and reloading about 11 ms: the
    gain is reproducibility, not speed.
- **2026-10-12**: pre-release audit, before R8, at the user's request. Three
  independent reviews ran: a new user following the docs on a clean install, a
  0.8-to-0.9 consistency check, and a bug hunt with reproductions. Each finding
  was reproduced before it was fixed.
  - **Silent wrong data, fixed:**
    - `overwrite` left a previous output's files;
    - outputs could replace raw channels;
    - NaN clocks were accepted;
    - a suffixed sub-channel ignored its directory's latency.
  - **Crashes and wrong answers, fixed:**
    - a non-numeric latency crashed status and check;
    - `filter()` had no `keys`, a bug from 0.8 that broke the README pipeline;
    - SemanticKITTI ignored `split=`;
    - `status -s` showed 0 Rellis labels;
    - the csv scan disagreed with the loader;
    - a BOM broke tables;
    - a plugin whose detect() raised broke every dataset;
    - views were stranded by aliases;
    - check refused sidecar keys on stacked formats.
  - The CHANGELOG gained an "Upgrading from 0.8" section and lost a false
    claim. The CLI docs, the schema table and a dozen statements in the guides
    were brought in line with the code.
  - **For R8:** `apairo_extractor` registers its preprocess outputs with
    `getattr(cls, "output_loader", "npys")`. It must fall back to `npys` when
    that is `None`, or merging `apairo_preprocess`'s
    `feature/output-format-from-input` breaks it. Its `export_mnt.py` also
    still imports `MNTDataset`, removed from apairo in 0.3.0.
- **2026-10-15**: tutorial session 1, a fresh install of 0.9.0 used as a
  newcomer would on the `barakuda1` extraction, added U1–U5. U1 landed on
  `feature/tutorial-s1-ergonomics` the same day (949 tests, ruff and mypy
  clean), with the merge pending.
- **2026-10-16**: U1 and `check` without `apairo init` merged into `main`
  (952 tests, ruff and mypy clean).
  - The two `apairo_extractor` gaps noted for R8 are fixed on its `main`
    (`a914697`): preprocess outputs are registered as written, `npys` per
    frame and `npy` stacked, and the MNT export keeps its own paths instead
    of importing `MNTDataset`. The direct bag-to-MNT path reads ROS2 bags too.
  - `release/0.9.0` (`9144c31`, 2026-10-15, unpublished) predates both merges.
    R8 cuts it again from `main`; the tutorial venv still runs that first cut.
- **2026-10-19**: U2 and U5 named `apairo_rr`, deprecated since 2026-07-23
  and folded into `apairo_visu` as its `rerun` extra; the local clone was
  behind. They now name `apairo_visu`. Its Rerun examples still said
  `pip install apairo-rr`, and now name the extra (`f65109b`).
- **2026-10-20**: U4 landed. The docs no longer call the per-channel
  directory layout KITTI, and `concepts.md` says which storage to choose,
  with `Convert` to move a channel between them; `IDEAS.md` holds the
  `RawDataset` name for 1.0. In `apairo_extractor` (`9365d84`), the export
  is "apairo", not "KITTI", in the TUI, `--help`, the README and the
  package description. The scope had suggested `files` and `zarr`, "both
  with the `.apairo` sidecar"; the second output is the MNT mission, which
  has no sidecar, so it keeps its name. No flag changed. What `grep -ri
  kitti` still finds is SemanticKITTI and KITTI's own conventions
  (`poses.txt`, frames numbered alike across sensors).
  - Also: the logo is redrawn solid, with a heavier wordmark (`fbc6174`),
    on the docs and the organisation site.
- **2026-10-23**: D1–D5 added: one docs site for the ecosystem, the
  satellites' docs built from their repositories, executed examples, a
  studio recording, then after JOSS a live studio and real-data results. U5
  also decides whether `apairo_transform` and `apairo_preprocess` publish as
  one package. The dataset API tab is now "User Guide", with a
  Synchronization page (`9a52ce2`); mkdocs validates anchors.
- **2026-10-23**: U2 landed. `installation.md` has an Ecosystem section and
  the README links to it. Each install line ran in a fresh Python 3.12
  virtualenv, against apairo 0.8.0 from PyPI: the extractor, transform and
  preprocess packages, then `apairo_visu` once per extra. `apairo studio`
  served `mini_tartan` (HTTP 200), `apairo rerun --save` wrote a `.rrd`, and
  open3d 0.20 imported; `apairo extractor` ran `--help` only, with no bag at
  hand. Found on the way, in `apairo_visu`: without `--lidar`, `apairo rerun`
  offers `multisense_imu` only, not `velodyne_0`: `rerun/cli.py` reads
  `ds.available_keys`, which does not exist, and falls back to the first
  event's channels, a single one on an asynchronous dataset. Use
  `ds.available`.
  Fixed the same day in `apairo_visu` (`a1e3eb3`). `cli.md` also opens on a
  typical session and lists the four ecosystem commands, each checked on
  `mini_tartan` except the extractor.
