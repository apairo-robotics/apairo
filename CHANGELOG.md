# Changelog

All notable changes to apairo are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/) — the public API and the on-disk
`.apairo` format stabilize at **1.0**.

## [Unreleased]

### Added
- **A new storage format is a plugin, not a core change.** Every channel is
  now read through one public contract, `apairo.core.formats.Format`: which
  directories hold the format, which files are its frames, the loader that
  decodes one, the clock forms its data provides, the channel fields it reads,
  what `status` shows and what `declare` suggests. A package registers one
  through the `apairo.formats` entry point group (or `register_format()` in
  process), and `loader: <name>` then works in `init`, `declare`, `check`,
  `status`, loading and `synchronize()`, with its own fields and `key` forms
  accepted by `check`. The seven built-in formats moved onto the same
  contract, and the read path no longer names a format (a test enforces it):
  the CLI, the configuration checks and the asynchronous dataset ask the
  registry. `apairo.testing.check_format` is the conformance check a plugin
  runs in its own tests. The new guide "Write a Format Plugin" builds a
  complete `.xyz` point-cloud plugin in about forty lines, and the test suite
  installs that plugin through its entry point and runs it end to end. A
  plugin that fails to import is skipped with a warning; a built-in name
  cannot be taken over.
- **A preprocess writes in any format, its input's by default.** The format
  contract covers writing too:
  - a format stores a frame (`write_frame`) or a whole channel
    (`write_channel`), says what it can hold (`can_write`), and must read back
    unchanged what it wrote, which `check_format` verifies;
  - `npys`, `npy`, `img` (PNG), `bin`, `zarr` and `csv` write, and a plugin
    can too;
  - an output goes in the format the run asks for
    (`run_preprocess(..., output_format=...)`), else the preprocessor's
    `output_loader` (now optional), else its input channel's format when that
    format can hold it, else `npys` / `npy`. An image mask stays an image, a
    crop of `.xyz` clouds stays `.xyz`;
  - a stacked format can take a frame preprocessor's output, stacked per
    sequence;
  - a format conversion is a preprocess: `Convert("velodyne_0", to="zarr")`
    writes a derived channel with its clock and provenance;
  - a format that cannot hold an output, or is read-only, is refused by name
    before anything is written.

  Existing preprocessors declare `output_loader` and write exactly as before;
  `apairo_preprocess`'s suite passes unchanged.
- **Standard datasets ship with apairo and install by name.**
  - `pip install apairo[tartan]` (also `rellis`, `goose`, `semantic-kitti`,
    `tum`, `euroc`) brings what reading that dataset needs. The install line
    stays the same if a dataset later moves to a package of its own.
  - The TUM RGB-D and EuRoC declarations moved from `examples/` into the
    package and are selected by name: `declare="tum_rgbd"`,
    `--declare euroc_vicon_room`.
  - `apairo init --as` takes its classes from a registry. `TartanKittiDataset`
    joins the list, and a dataset kept in its own package registers through
    the `apairo.datasets` entry point group.
  - Adding a standard dataset is a pull request: a declaration or a profiled
    class, a miniature fixture, a guide and an extra ("Adding a Dataset").
- **A synchronization can be persisted, and reloaded as the same frames.**
  `synchronize(...).persist(name)` writes what the view is into the
  sequence's `.apairo`, with no data copied: the kept reference ticks, one
  index array per channel, the method, the tolerance, and a fingerprint of
  every source channel (its frame count and a hash of its clock).
  `RawDataset(seq).load_view(name)` rebuilds the same synchronous dataset from
  those indices without matching again. It refuses a view whose sources
  changed since -- a frame added, a timestamp moved, a `latency` declared, or
  another declaration -- and names the channel. On a root, each sequence keeps
  its own view. An interpolated channel takes its interpolator again at load
  (it is code, not state); a custom matcher is not needed. `apairo status`
  lists a sequence's views, `apairo check` validates `views.yaml`, and the
  schema documents it. On a real TartanDrive sequence, the view of 5041 frames
  over three channels is a 162 KB index file. Its frames reload identical.
- **`latency`: a sensor that stamps its readings late, corrected by
  declaration.** A channel field in seconds -- how long before its timestamp
  each frame was captured -- that moves the channel's clock back, whatever its
  source (a `key`, a `timestamps.txt`, a borrowed clock), before
  `synchronize()` aligns it. `status` shows the corrected span, and `check`
  wants a number.
- **Guide: the KUKA F/T and IMU logs, a robot arm read in place.** Three
  sensors, each with its own CSV file and microsecond clock: orientation at
  100 Hz, force/torque at 700 Hz, IMU at 254 Hz. The `kuka_ft_imu`
  declaration ships with apairo and declares the IMU's documented 8416 µs
  delay. The example (`examples/kuka_ft_imu.py`) finds that delay in the
  data, by fitting the IMU's acceleration to the force while the arm rotates.
  The IMU shows up 7.6 ms late on the logged clocks, and −0.8 ms after the
  correction. It then aligns both sensors onto the robot's clock. CI runs it
  on a real excerpt of the record (`mini_kuka_ft_imu`, CC BY 4.0); the guide's
  figures come from the full record.
- **Every standard dataset is checked on a sample of it, against one
  contract.**
  - `apairo.testing.check_dataset` opens a sample as a user would, with
    `init` then loading. It checks:
    - that every channel's frames keep their dtype, rank and shape;
    - that the clock is finite, has one timestamp per frame and never goes
      back;
    - that `apairo check` reports nothing;
    - that the splits do not overlap;
    - that a preprocess output writes and reads back;
    - what the sample's card expects: shapes, dtypes, label ranges.
  - Each sample is described by a card in `test/assets/samples.yaml`: its
    source, licence and expectations. CI fails when a standard dataset or a
    shipped declaration has no sample.
  - `pytest -m realdata` runs the same checks on full copies named by
    `APAIRO_REAL_*` variables, through a stand-in of links, so nothing is
    written next to the data.
  - New real samples: `mini_goose`, 9 scans of the GOOSE archives, and
    `mini_semantic_kitti`, real SemanticKITTI labels with synthetic clouds,
    since KITTI's scans need a registration. Both are rebuilt by
    `fetch_public_samples.py` from the official archives with HTTP range
    requests.
  - A dataset package runs the same check in its own tests.
  - The check validates a shipped declaration against the sample, as
    `apairo check --declare` does.
- **`csv` loader: a text table is a channel, its clock a column.** IMU logs
  and ground-truth trajectories in most SLAM datasets are one table with one
  row per frame and the timestamp in a column -- EuRoC's `imu0/data.csv`, TUM
  RGB-D's `groundtruth.txt`. `loader: csv` reads it in place (comma, tab or
  whitespace separated, `#` comments, names from a header row or from EuRoC /
  TUM style header comments, trailing `[unit]` dropped), and the new
  `key: {column: <index or name>, units: [ns]}` form takes the clock from a
  column, keeping it out of the frame data. An integer nanosecond stamp is
  converted exactly before scaling. `fields` selects columns by name and
  `array_file` names the table (a `.txt`, or one of several); `directory: "."`
  reaches a table at the sequence root. `init` detects a `.csv` directory,
  `status` reports its rows, rate and width, and `check` validates the column
  key.
- **A directory that holds its data files itself is a dataset.** `init`
  answered "no channels" for a channel directory opened on its own
  (`seq/velodyne_0`) and for a logger's folder of CSV tables. A directory with
  no sub-directory is now a sequence of its own: its frames are one channel
  named after the directory (with its suffixed variants), and each table is one
  channel named after its file -- or after the directory when it is alone --
  all reading from `directory: "."`. Loading, `init` (and `--merge`), `status`,
  `check` and `declare` accept both entry points; a directory with a
  sub-directory keeps its meaning, so a table beside channel directories is
  not picked up on its own.
- **`declare` suggests a table's clock column.** For a `csv` channel with no
  `timestamps.txt`, the column whose header names a clock (`timestamp`, `time`,
  `t`, ...) and never decreases becomes `key: {column: ...}`, its unit guessed
  from the width of an epoch. On a real KUKA recording (three CSV logs at 100,
  254 and 698 Hz), `init`, `declare` and `status` now give every rate with no
  edit.
- **Dataset guides: TUM RGB-D and EuRoC MAV, read from a declaration only.**
  Two docs pages walk through a well-known SLAM dataset loaded in place by
  `RawDataset` -- no subclass, no conversion: where each clock lives (image
  filenames in seconds or nanoseconds, a column of a text table), the
  `apairo.yaml` that says so, and a `synchronize()` call with what each
  match's offset tells you. The declarations ship with apairo, by name
  (`tum_rgbd`, `euroc_vicon_room`, `euroc_machine_hall`), with
  two runnable examples, `tum_rgbd_associate.py` and `euroc_synchronize.py`.
  All three were checked against real sequences (`freiburg1_xyz`,
  `V1_01_easy`, `MH_05_difficult`); CI runs the examples on synthetic
  miniatures of the layouts (`test/assets/mini_tum`, `mini_euroc`, generated
  by `make_slam_fixtures.py`). The TUM guide compares `synchronize(nearest)`
  with TUM's own `associate.py` on a real sequence: 789 of its 792 pairs are
  identical, and the difference -- per-frame matching against one-to-one --
  is stated rather than glossed over.

### Changed
- **`status` counts what loading reads.** A channel's frame count, shape and
  dtype now come from its format, over the same frame files loading resolves
  (a `key`/`order` regex, a suffixed variant), rather than from the length of
  its `timestamps.txt`; a mismatch between the two is still reported by
  `check`. The shape is read off the frame loading calls frame 0: on an
  export of clouds named `<scene>_<stamp>.pcd`, `status` showed the
  alphabetically first file (45658 points) where loading starts at the
  earliest stamp (48533). A `zarr` channel now shows its shape. On the TUM,
  EuRoC (both halls), KUKA, TartanDrive and evaluation exports at hand,
  `status`, `check` and `declare` are otherwise unchanged, byte for byte.
- **`declare` gives a stacked array a sidecar hint.** A lone `.npy` without a
  `timestamps.txt` used to get a filename `key` regex, which `check` then
  refused on a stacked loader; it now gets a commented `key: {file: ...}`.

- **A profiled dataset reads its derived channels through their format.**
  SemanticKITTI, Rellis, GOOSE and the other profiled datasets read a
  preprocess output by its file extension. A derived `bin` channel came back
  flat instead of `(N, 4)`, and a derived image channel was not found. It is
  now opened by its format, so every format that writes reads back there too,
  per frame or stacked per sequence. Their raw modalities are unchanged: their
  YAML profiles describe them.
- **Overwrite protection looks at the output channel's directory.**
  `run_preprocess` refuses to write into a channel directory that already
  holds data, whatever the format; it used to look for the first frame file.

### Removed
- `apairo.core.config.KNOWN_LOADERS`: the registry is the list of formats,
  `apairo.core.formats.format_names()`. `apairo.loader.str_to_loader` stays, as
  a read-only view of the registry.

### Fixed
- **GOOSE labels are the class, and GOOSE has a clock.** A GOOSE label holds
  the class in its lower 16 bits and the instance in the upper 16, like
  SemanticKITTI, but the profile did not mask it. On real scans, 13 to 20% of
  the points came back as values up to 18,939,940 instead of a class from 0 to
  59. The tests drew labels from 0 to 63, so they could not see it; the real
  sample's expected range does. The profile also reads each scan's capture
  time from its file name (`<scene>__<frame>_<ns>_vls128.bin`), which GOOSE
  had been loaded without. A tree in the GOOSE format whose files are named
  otherwise stays clockless, as one without its clock source does.
- **A clock must hold one timestamp per frame.** The timeline gives a channel
  one slot per timestamp, and nothing checked that this matched its frames: a
  `timestamps.txt` one line short silently dropped the last frame (a gap in
  the middle shifted every later frame onto the wrong timestamp), while a long
  one -- or a `timestamps_from` borrowed from a channel with more frames --
  built fine and died later on a bare `IndexError`. Loading now refuses such a
  channel at construction, naming both counts and where the clock came from.
  Checked on every apairo dataset at hand before landing: none was affected.
- **`check` is quiet on what apairo's own tools write.** Every dataset from
  `apairo_extractor` was reported for `transform: unknown field 'source'` --
  the extractor records the TF topic a transform was read from. `source` is now
  a v1 transform field (provenance, descriptive like the rest). A sidecar
  written before 0.2.1 carried `has_timestamps` on every channel, one warning
  each; it is now reported once per file as deprecated, naming the channels
  and saying to delete the lines. On the local barakuda extractions, `check`
  goes from two warnings to none, and from two to one notice.
- **An index file is not a channel.** Since the `csv` loader, any directory
  holding a `.csv` was detected as a table channel -- EuRoC's `cam0/`, whose
  `data.csv` pairs each stamp with an image filename, became a channel that
  could not load. A `.csv` (or `.txt`) is now detected only when its first rows
  read as a numeric table.
- **`status` shows a suffixed variant's own shape, and a sparse rate.** A
  suffixed channel (`velodyne_0_intensity`) showed the shape of the base
  frames beside it; it now reads its own files. A channel with a frame every
  few seconds printed `0.0 Hz`; it prints `0.021 Hz`.
- **A preprocessor with several inputs runs on an asynchronous dataset.**
  `run_preprocess` iterated the interleaved event timeline, one channel per
  sample, so a preprocessor with two or more `input_keys` died on its first
  frame with a `KeyError` -- every multi-input preprocessor was limited to the
  synchronous profiled datasets. When the inputs share one clock (identical
  timestamps: a channel derived through `timestamps_from`, or derived channels
  with equal `timestamps.txt`), the runner now zips them row for row, for
  frame and sequence preprocessors, on a sequence or a root; the output is
  numbered by row and stamped by that clock. Inputs on different clocks are
  refused before anything is written, naming each channel and its frame
  count: pairing them is a synchronisation, which the runner does not guess.
  Checked on a real TartanDrive sequence with `GroundHeightFromLabels`
  (voxelised cloud + ground labels), which failed on `KeyError:
  'ground_ransac'`.
- **`status` and `check` read the layout through `--declare`.** An external
  declaration was only validated, not applied: a channel whose clock it
  declared showed up without its rate, a table channel it added did not show
  at all, and `check` -- since it reports clockless channels -- called a
  dataset that loads fine with `declare=` broken. Both now merge the file
  last, as loading does, so they describe what `RawDataset(..., declare=...)`
  sees. Found while checking the TUM declaration against a real sequence.
- **`status` shows the rate and the shape of every channel.** A channel whose
  clock is a declared `key` -- parsed from its filenames, or read from a named
  sidecar -- printed `-` for its rate and span, because only a
  `timestamps.txt` was read; `img`, `bin` and `pcd` channels printed `?` for
  their shape, because only `.npy` headers were. The key is now parsed the
  way loading parses it (and ahead of a `timestamps.txt`, as loading does),
  and the shape and dtype come from the channel's first frame. Seen on the TUM
  and EuRoC guides, whose image channels now read `30.0 Hz` and
  `(480, 640, 3) uint8`.
- **`check` sees what loading refuses.** It now reports, without loading
  anything, a channel whose clock does not match its frame count and a
  channel with no clock at all (no `timestamps.txt`, `key` or
  `timestamps_from`) -- both used to pass as `OK -- no issues`.

## [0.8.0] - 2026-09-02

### Added
- **A dataset root's `apairo.yaml` applies to every sequence.** The root file
  is the dataset-wide contract -- typically the union of the sequences'
  channels -- propagated below each sequence's own `apairo.yaml` (per-field
  precedence: `declare=` > sequence file > root file > registry). In
  load-everything mode a sequence skips declared channels it does not hold
  (an explicitly requested missing channel still errors), and `available`
  reflects what is actually on disk. `apairo declare <root>` now writes
  `<root>/apairo.yaml` (one scaffold, every sequence) instead of requiring
  `-o`, and `apairo declare --help` explains the contract, the naming rule
  and the examples. A sequence opened **standalone** inherits its parent's
  `apairo.yaml` automatically (one-level upward look), so entering a dataset
  by a sequence -- `RawDataset(root/"seq")`, `apairo studio root/seq` -- reads
  the same contract as entering by the root; `status` and `init` inherit the
  same way.
- **`CITATION.cff`.** The repository declares how it wants to be cited, so
  GitHub shows a "Cite this repository" box and reference managers read the
  entry directly. The DOI field is filled from the Zenodo archive of this
  release.

### Fixed
- **An empty `channels.yaml` no longer crashes every command.** A truncated or
  stray whitespace-only sidecar (seen on a shared vault) made `yaml.safe_load`
  return `None` and every reader die on `.get` -- `read_config` and
  `load_profile` now read it as `{}`, so `status`/`check` report the malformed
  file as an issue instead of a traceback.
- **`apairo status` now sees what loading sees.** The in-tree declaration is
  merged over the registry before rendering, and a channel enumerated by a
  `key`/`order` regex counts only the loader-extension files whose stem
  matches — colocated schema yamls or archives no longer inflate `frames`
  (a 6-label channel used to report 13). The scaffold's `key:` hint also got
  honest: it inspects only the loader's own files (a stray `.tar.xz` no longer
  hijacks the stem inspection), keeps a literal tail after the epoch in the
  suggested regex (`(\d+)_toaster$`), and is emitted **uncommented** when the
  channel is unloadable without it and the guess is confident — a scaffold
  should load as generated.
- **`directory:` is now honored for plain channels, including nested relative
  paths.** The schema always described `directory` as "the on-disk subdirectory
  the channel's files live in", but the implementation only consumed it for
  `suffix`/`array_file` colocation — on a plain channel it was silently
  ignored. It now points any channel at its directory: another top-level one,
  or a nested relative path (`directory: per_object_gt/pcd` — an annotation
  tool's export tree), resolved from the sequence directory and validated
  against `..`/absolute escapes. No more symlinking nested exports to the top
  level just to load them.

### Added
- **`apairo.yaml` — the human-owned declaration.** Channel metadata used to
  have a single home, `.apairo/channels.yaml`, which mixed two owners: what the
  machine discovers and derives (`kind: raw` scan results, `kind: preprocess`
  provenance) and what only a human can know (a filename `key` regex, a `pcd`
  `fields` contract, an `alias`). That made `init --overwrite` destructive to
  hand-written knowledge, and made the read-time `key` feature — whose promise
  is *zero writes into your data tree* — require a write into that very tree.
  The split is by owner, not by content: `<seq>/apairo.yaml` (visible,
  git-versionable, **never written by apairo**) declares *how to read* a
  directory, while `.apairo/` remains the machine registry of *what exists*.
  A declaration uses the `channels.yaml` version-1 schema minus machine
  provenance (`kind: preprocess`, `sources`, `recipe` are refused — loudly).
  It overlays the registry per channel *and per field*, so declaring a `key`
  does not repeat the loader the registry already knows; a declaration-only
  channel must name its `loader`. `RawDataset(dir, declare=path)` loads a
  declaration from *outside* the tree (precedence: `declare=` >
  `<seq>/apairo.yaml` > `.apairo/channels.yaml`), making fully-featured loading
  of a read-only mount possible; a root propagates `declare=` to every
  sequence. The bootstrap scan respects declarations: a channel whose stems a
  declared `key`/`order` regex explains is not fanned out into spurious
  suffixed sub-channels. `apairo status` validates the in-tree declaration
  (ownership violations, unknown fields, loaders, `key`/`order` specs).
  The in-tree file needs no flag anywhere — its name is a fixed convention,
  auto-discovered like `.apairo/` itself; external files reach the CLI via
  `apairo init/status/check --declare FILE` (respect during the scan,
  validation against the dataset) and the `declare=` parameter on
  `RawDataset.init`. Purely additive: without a declaration nothing changes.
- **`apairo declare` — scaffold a declaration.** The human counterpart of
  `init`: scans a directory and writes a starter `apairo.yaml` (or `-o FILE`,
  `-` for stdout) with every detected channel and its loader, plus commented
  hints — the actual field list read from the first frame's PCD header, and a
  `key:` suggestion with the unit guessed from a trailing digit run in the
  stems (a 19-digit epoch suggests `units: [ns]`). Refuses to overwrite an
  existing declaration; on a root it requires `-o`, since a root-level
  `apairo.yaml` is not auto-discovered.
- **`pcd` — PCL point clouds read in place.** A directory of vendor `.pcd`
  frames is now a channel like any other (`loader: pcd`), for PCD v0.7 `ascii`
  and `binary`. No loader could express this before: `bin` assumes a headerless
  float32 quadruple and would have read the ASCII header as coordinates, and a
  PCD→npy preprocessor would still need a parser while materialising a second
  copy of the vendor's data. No new dependency — numpy alone.
  A PCD header is self-describing, so the field set is a *per-file* property.
  A channel pins it with `fields: [x, y, z, intensity]` in `channels.yaml`: that
  selects those columns in that order, so the channel's width is a declared
  property rather than a per-file accident, and a frame missing one raises
  naming both sets. Without `fields`, every field the file declares is returned
  in header order. The field contract lives in the layout, never in the loader.
  `COUNT > 1` expands to `<name>_<i>` columns; PCL's `_` padding fields are
  dropped but still consume their slot; the result dtype is the smallest type
  holding every selected field exactly (`np.result_type`), so an all-`F4` cloud
  stays float32 while an Ouster `t` (`U4`) promotes to float64 instead of losing
  counts to a float32 mantissa. `DATA binary_compressed` is refused by name
  (it needs an LZF decompressor) rather than mis-parsed into plausible garbage.
  Auto-detection maps a `.pcd` directory to the loader, so `RawDataset.init`
  registers such a channel; a filename `key`/`order` regex works with it.
  Purely additive. Validated against 45 real vendor clouds, bit for bit.

## [0.7.0] - 2026-07-23

### Added
- **Synchronous datasets can carry a per-frame clock; `is_synchronous` is now
  structural.** Synchrony (co-captured frames — `ds[i]` is one sample across all
  channels) and the presence of a timestamp are orthogonal: a synchronous
  dataset may expose a shared per-frame clock in `ds.timestamps` (so
  `sample.timestamp` is that frame's tick), while `is_synchronous` is backed by
  an explicit structural flag rather than `timestamps is None`. The clock's
  origin is a dataset concern, resolved most-specific first: a subclass
  `_clock_provider` callable, a profile-level `clock:` (`{dir, name/units, ext}`
  self-contained, a per-sequence sidecar `{file}`, or `{channel: X}`), or an
  in-band channel `key:`. A profile `clock:` aligns to the selected frames by
  `(sequence, row)`, so the clock source need not be among the loaded keys and
  still lines up under any split; its clock resets per sequence and is validated
  per sequence. Rellis-3D derives its clock from the co-captured camera filenames
  (without loading the camera) and SemanticKITTI from each sequence's `times.txt`.
  Purely additive: a dataset that declares no clock, or whose declared source is
  absent, stays clockless (`timestamps is None`).
- **`array_file` — colocated stacked-`npy` channels.** A directory holding two
  whole stacked arrays (e.g. `gicp_poses/poses.npy` beside `valid_mask.npy`, one
  row per pose) can now expose each as its own `npy` channel: declare
  `array_file: <name>` to name the exact file, and `directory:` to share another
  channel's directory — the whole-array analogue of the per-frame `suffix`
  idiom. Without it, `npy` still loads the sole `.npy` in the directory
  (unchanged). The filename lives in `channels.yaml`, never in the loader;
  `verify_config` checks the file exists and that `array_file` is used only with
  the `npy` loader; `apairo status` resolves the shared directory. Purely
  additive. Auto-detection of such a directory is unchanged (still declare it
  explicitly); no apairo writer yet emits a colocated pair.

### Fixed

An audit (a self-review of the clock/`array_file` changes plus a broad
multi-dimension pass) fixed 35 defects across correctness, crashes, security,
docs and performance.

- **Silent data corruption**
  - Profiled datasets sorted per-frame files lexicographically and used that
    position as the row index into stacked per-sequence files (poses) and sidecar
    clocks (`times.txt`); non-zero-padded names (`0, 1, 10, 2, …`) silently
    misaligned. Discovery now uses a natural (numeric) sort, matching the async
    family — identical for zero-padded names.
  - Stacked loaders (`NPYLoader` / the new `array_file`, `_StackedSequenceLoader`)
    returned views into their persistent cache, so an in-place transform corrupted
    the dataset and two reads aliased; they now return copies, like the per-frame
    loaders.
  - `cache()` dropped the parent's sequence provenance, so `cache().window()` built
    windows across sequence boundaries; `CachedDataset` now snapshots
    `frame_sequence_ids` / `frame_stems` / `frame_channel_ids` / `timestamps`.
  - `SequencePreprocessor` on an asynchronous `RawDataset` root ignored sequence
    boundaries (only `ProfiledDataset` had `_seq_groups`); the root now exposes
    per-sequence groups, so a global preprocessor runs once per sequence, writes
    one output per sequence, and no longer registers a channel whose file exists
    only in the first sequence.
- **Crashes on valid input**
  - `ConcatDataset` misread a synchronous dataset that carries a per-frame clock
    (reported it asynchronous, `IndexError`'d indexing an ndarray clock by channel
    name, and `AttributeError`'d on view children); it now uses the
    `is_synchronous` protocol. `ZipDataset` gained a `timestamps` attribute.
  - A profile `clock:` with no per-frame anchor (`keys=['poses']`) or partial
    per-sequence coverage now stays clockless instead of aborting construction.
  - A `timestamps_from` source whose clock is filename-parsed no longer crashes
    order-dependently — the source's clock is resolved through the same precedence
    (provider / `key` / `timestamps.txt`).
  - `bin` / `zarr` / `npy` channels with a `key` / `order` regex no longer raise a
    bare `TypeError`: `BINLoader` accepts frame-ordered `files=`, and a filename
    key/order on a stacked loader is rejected clearly at `verify_config` time.
  - `synchronize()` no longer collapses to a single frame when a channel has one
    event (a NaN frequency was picked as the reference clock).
  - `IMGLoader` matches `png` / `jpg` / `jpeg` / `bmp` case-insensitively and sorts
    numeric-first with a lexicographic fallback (was case-sensitive `png` / `jpg`
    plus an `int()` sort that crashed on timestamped names).
  - `read_calibration`, `get_end_of_time`, and `BINLoader` / the profiled per-frame
    reshape now fail with a clear entry/channel/file-named error instead of a bare
    `KeyError` / `IndexError` / reshape `ValueError` on a malformed input.
  - `assert`-based invariants in `derived_path` / `_full_anchor_rows` are now
    explicit raises (they survive `python -O`).
- **Security** — a config-supplied path (`directory`, `array_file`, `clock: {file}`,
  a sidecar name) from an untrusted `channels.yaml` / profile is validated: absolute
  paths, drives, root anchors and `..` escapes are rejected under both POSIX and
  Windows rules, so a downloaded dataset cannot read or write outside its own tree.
- **`export()`** — `overwrite=True` now replaces the destination (clears it first)
  instead of silently merging stale sequences into the regenerated manifest, and a
  channel selected by its real name while carrying an alias is no longer dropped.
- **Validation & docs** — `verify_calibration` accepts a null `D` / `R` / `P`;
  `verify_config` no longer flags a self-alias; `_verify_key_order` tolerates a
  non-string `units`. `filter_split()` now supports directory-layer splits (GOOSE),
  and the `filter` / `split` contracts are documented (filter reads the stored
  channel before transforms; `split` re-instantiates, `filter_split` preserves
  transforms). Docs corrected to the numpy (not `torch.Tensor`) sample contract,
  dropping the nonexistent `TartanDataset` / `Torch*Dataset` / `.pt` entries and the
  phantom PyTorch dependency and `apairo[viz]` extra.
- **Performance** — `_full_anchor_rows` is memoized and `frame_sequence_ids` /
  `frame_stems` are cached, avoiding repeated full-tree globs and O(n) rebuilds
  during construction.

## [0.6.2] — 2026-07-21

### Added
- **`key: {units: [...]}` — readable sugar for `scale`.** When every capture
  group of a filename `key` regex is a time field, name the units
  (`s` / `ms` / `us` / `ns`) instead of raw multipliers: `units: [s, ms]`
  compiles to `scale: [1, 0.001]`, self-documenting and steering off the
  default-join footgun for non-zero-padded fractional fields. `units` and
  `scale` are mutually exclusive; unknown units and a length ≠ capture-group
  count are flagged by `verify_config` and at load. Documents the heterogeneous
  `camera_<sec>_<index>` pattern: capture only the key field and let `order` sort
  by the index (an index is not a duration, so it never folds into the key).

## [0.6.1] — 2026-07-21

### Added
- **`run_preprocess(..., reuse=True)` — recipe-addressed idempotency.** Each
  materialized channel now records a `recipe` hash of its producing
  preprocessor's *declared* config (class, declared I/O, scalar constructor
  params -- never code). With `reuse=True`, re-running a preprocessor whose
  output is already on disk under an identical recipe is a no-op, and a *changed*
  scalar parameter regenerates it -- so a branching experiment reuses unchanged
  derived channels and never silently loads output from a different recipe. The
  default is unchanged (an existing output still raises without `overwrite`); the
  new `recipe` sidecar field is additive and provenance-only.

## [0.6.0] — 2026-07-21

### Fixed
- **Loading raw data never requires write access.** Constructing a
  `RawDataset`/`TartanKittiDataset` on a bare tree bootstraps the `.apairo`
  sidecar; on a read-only directory (shared cluster mount, ro container
  volume) that write raised `PermissionError` and the load failed. The
  bootstrapped config now falls back to memory with a warning -- run
  `apairo init` on a writable copy to persist it. The committed test fixtures
  stay bare (`test/assets/**/.apairo/` is ignored) so the smoke tests keep
  exercising the bootstrap path.
- **Datasets with transforms are picklable.** The pipeline steps built by
  `transform(key, fn)` and `transform(preprocessor)` were local closures, so
  any dataset carrying one could not be sent to spawn-based `DataLoader`
  workers (macOS/Windows default). They are now small module-level step
  objects; picklability is locked by a test and by the soak.

### Added
- **Filename-encoded and sidecar keys — an async channel supplies its own
  alignment clock.** A channel in `channels.yaml` can now declare
  `key: {name: <regex>}` to parse its alignment key from each filename's stem
  (one capture group → an integer index, two → `<sec>.<frac>`, or an explicit
  `scale: [...]` unit combine), or `key: {file: <name>}` to read it from a named
  sidecar — generalizing the hardwired `timestamps.txt`. A separate
  `order: {name: <regex>}` sets the enumeration policy (defaulting to the key's
  regex, else the frame-file convention), so filenames carrying a `_` the default
  convention rejects — a Rellis `<epoch>_<ms>` stem — still load. Keys are
  computed in memory at read time; **nothing is ever written** into the source
  tree, so a filename-keyed channel needs no sidecar at all. The parsed key feeds
  `synchronize()` unchanged (nearest/previous/next, tolerance, sparse subsets
  included). Both fields are additive and opt-in — a channel with neither keeps
  today's behavior (`timestamps.txt` / `timestamps_from` / frame position). A
  subclass can override either with `self._key_providers[channel]` /
  `self._order_providers[channel]` callables set before `super().__init__()`.
  Lands the Rellis-3D camera (2847 frames @ 10 Hz plus 1200 sparse image-labels)
  in two lines of `channels.yaml` — ends the "transcode filenames into a
  `timestamps.txt` before apairo can read the channel" workaround. See
  [Bring your own dataset](docs/datasets/bring-your-own-dataset.md).
- **`export()` — materialize a dataset subset to a new self-contained root.**
  `RawDataset(root, keys=[...]).filter_sequences([...]).export(dest)` (and
  `apairo export <src> <dest> --keys ... --sequences ...`) copies a structural
  subset -- whole sequences × whole channels of the asynchronous family -- to a
  fresh root, regenerating the `.apairo` sidecars so the copy is self-contained
  (`apairo status` on it reports exactly the exported channels, unlike an
  `rsync` that leaves them stale). Each channel keeps its own `timestamps.txt`
  and its provenance (`timestamps_from`/`sources`) trimmed to the subset,
  `calibration.yaml` is copied verbatim, third-party sidecars are dropped, and
  `--link` hardlinks data files on the same filesystem (near-free). Re-clocked
  (`synchronize`), cached, windowed or frame-filtered views are rejected -- the
  materializing export stays future work (see `IDEAS.md`). Ends the "rsync a
  subset and get stale sidecars" workaround.
- **Multi-output preprocessors.** A preprocessor can declare
  `output_keys: list[str]` instead of `output_key` (exclusive, validated at
  class definition) and return a `dict` with exactly those keys; one pass
  then materializes N derived channels. `run_preprocess` writes one channel
  per key (same `output_loader`) and registers all of them in `.apairo` with
  shared `timestamps_from`/`sources` provenance, so each stays individually
  selectable. Overwrite protection checks every declared key. The lazy
  preview (`transform(preprocessor)`) publishes every key of the dict;
  `output=` renaming and `keep=False` apply to the whole set (rename is
  rejected, drop covers all keys). Ends the "run the expensive voxelization
  N times or hand-write channels.yaml" workaround.
- **Provenance flows through `ConcatDataset` and `SynchronizedView`.** The
  `frame_info` / `frame_sequence_ids` / `frame_stems` contract, already
  forwarded by `FilteredView`/`ChannelView`/`WindowView`, now covers the two
  remaining views. `ConcatDataset` concatenates its children's ids and stems
  and dispatches `frame_info` to the owning child (ids are forwarded verbatim
  -- two children exposing the same id stay indistinguishable). A
  `SynchronizedView` carries the parent's sequence id when the parent belongs
  to a single sequence; since a root synchronizes per sequence and concats,
  `RawDataset(root).synchronize(...)` now reports sequence identity
  end-to-end. Both keep the availability probe: `AttributeError` when the
  underlying dataset exposes none (or, for a directly-built view, spans
  several sequences). Pure index arithmetic, no data reads.
- **Camera intrinsics in the core calibration.** `calibration.yaml` gains an
  additive `cameras:` section (one entry per physical camera, keyed by its
  frame -- the `CameraInfo` `frame_id`; field names mirror `CameraInfo` so an
  extractor writes them near-verbatim). `ds.calibration.get_intrinsics(cam)`
  returns a `CameraIntrinsics` (`K` 3x3, `distortion`, `model`, size, optional
  `R`/`P`); `register_intrinsics(...)` is the write-side sibling of
  `register_static_transform`, `verify_calibration` validates the section, and
  a root merges per-sequence cameras like it merges extrinsic edges. Same
  split as `get_tf`: the core stores and exposes, applying (projection,
  undistortion) stays in `apairo_transform`. Fixed along the way:
  `register_static_transform` rewrote `calibration.yaml` with only
  `{version, transforms}`, silently dropping any other section.
- **`transform(..., in_place=False)` -- branch instead of mutate.** The
  default stays in place (the statement idiom `ds.transform(...)` keeps
  working); `in_place=False` leaves `self` untouched and returns an
  independent branch -- a lightweight copy sharing loaders and indices but
  owning its pipeline. This closes the `v1 = ds.transform(a); v2 =
  ds.transform(b)` trap where both ended up stacked on the same object.
- **`benchmarks/soak.py` -- intensive-usage soak on synthetic data.** One
  session exercising the whole surface end to end (bootstrap, reload,
  timeline scan, `run_preprocess` persistence, lazy preview, synchronize,
  filter/select/cache/join/window, shuffled epochs, pickle roundtrip), every
  step asserting its contract. Runs in CI at small scale and via `make soak`;
  scale it with `--sequences/--frames/--points`. Complements `bench.py`
  (cost) with a correctness soak. Its first run caught the pickling bug above.
- **PEP 561 `py.typed` marker** -- the type annotations are now visible to
  consumers' type checkers.
- **mypy at zero errors, gated in CI.** The 74-error internal baseline is
  fixed for real (attribute contracts declared on the base class and the
  root mixin, `Literal` for `join(on_collision=)` and precise `FilteredView`
  returns, honest `Path`/`Optional` narrowing in the profiled layout), with
  10 documented `# type: ignore[code]` escapes where the pattern is
  intentionally dynamic (deprecation alias, duck-typed stream loaders,
  property-over-attribute overrides). Along the way `AsyncLayoutDataset.init`
  now returns the written `channels.yaml` path (previously `None`), and two
  malformed-profile cases fail with a clear message instead of an opaque
  `TypeError`.
- **Python 3.13 and 3.14** -- declared in the classifiers and tested in CI,
  alongside new macOS and Windows jobs. Dependency floors are now explicit
  (`numpy>=1.26`, `PyYAML>=6.0`); a dedicated CI job runs the suite against
  the numpy floor.

### Changed
- **CI and lint hardened toward 1.0.** ruff bumped 0.4 -> 0.15 with an
  explicit ruleset (defaults + import sorting `I`, modern syntax `UP`,
  bugbear `B`) and the whole tree reformatted with `ruff format` (now
  enforced with `--check` in CI); coverage is measured with a fail-under of
  85% (currently ~89%). Legacy test helpers writing to a CWD-relative `tmp/`
  (`test/paths.py`) were removed.

## [0.5.0] - 2026-07-06

### Fixed
- **`run_preprocess` on a multi-sequence `RawDataset`/`TartanKittiDataset` root.**
  It crashed in `derived_path` with `AttributeError: _sequence_dir` (a root has
  no single sequence dir). Each frame is now routed to the sub-sequence it
  belongs to via a shared `_locate()` helper on `RootSequenceMixin`, so per-frame
  outputs are written under the right `<sequence>/<key>/` directory, and the
  derived channel is registered in *every* sequence's `channels.yaml` (a single
  root-level registration was unloadable). Reloads at both the sequence and root
  level. Single-sequence and profiled-root datasets are unchanged.
- **`Sample.timestamp` contract corrected.** The docstring claimed
  "synchronous datasets: timestamp is None", but a `synchronize()` result is
  synchronous *and* carries the reference-clock tick (load-bearing:
  `run_preprocess` reads `sample.timestamp` to emit `timestamps.txt`). The
  contract is now stated per *clock*: async event -> its own timestamp;
  synchronous clocked frame (`synchronize()`) -> the reference tick; synchronous
  clockless frame (profiled dataset) -> `None`. No behaviour change.
- **`SynchronizedView.frame_info` documented as composite.** On a synchronised
  frame every channel is backed by a different source event, so `frame_info`
  reports `channel=None` and `row` is the **view index**, not an on-disk row.
  Per-channel provenance lives on `frame_indices`
  (`frame_indices[reference][idx]` is the clock event the tick was resampled
  onto). Behaviour unchanged; the previous fallback docstring implied a
  non-existent single origin.

### Added
- **`ds.transform(preprocessor)` -- lazy preview of a preprocess.** A
  `FramePreprocessor` is now a callable on a `Sample` (same protocol as
  transforms and `Interpolator`), so the same instance runs in both worlds:
  lazily in a pipeline (result published under its `output_key` at access
  time, nothing written -- iterate and visualize before committing) and
  materialized via `run_preprocess` once satisfied. `output=` overrides the
  published key; `keep=False` drops it from the final sample. A
  `SequencePreprocessor` is rejected with a pointer to `run_preprocess`
  (global context cannot run lazily). A missing declared input raises a
  clear `KeyError` instead of failing downstream.
- **`method="next"` matching strategy for `synchronize()`** -- first event with
  `t >= t_ref`, the forward counterpart of `"previous"`. The three built-in
  strategies now name their temporal direction: `"previous"` (past only),
  `"next"` (future only), `"nearest"` (either side, ties favour the earlier
  event).
- **`ds.window(size, stride=1, reduce=, boundary="clip")` -- temporal windowing
  as a lazy view.** Groups each frame with its `size - 1` causal neighbours
  (spaced `stride`), ordered oldest -> newest, and reduces them to one sample via
  the required `reduce` callable (`list[Sample] -> Sample`). Membership is index
  arithmetic computed at construction; windows never cross a sequence boundary
  (`frame_sequence_ids`, with a single-sequence fallback when absent, e.g. after
  `synchronize()`). `boundary="clip"` shrinks windows at sequence starts (one
  output per frame); `"drop"` keeps only full windows. This is the random-access
  counterpart to the stateful `AccumulateFrames` transform -- correct under
  `split`, shuffling and multi-worker `DataLoader`. Exposed as the chainable
  `AbstractDataset.window(...)` and the `WindowView` class.

### Deprecated
- **`Preprocessor.process()` in favour of `__call__`.** Subclasses should
  implement `__call__`; a legacy `process` is aliased to `__call__` with a
  `DeprecationWarning` at class definition, and calling `.process(...)` on a
  new-style instance warns and delegates. The runner now invokes the
  instance directly.

### Changed
- **`synchronize(method="latest")` renamed to `method="previous"`.** The old
  name did not say which direction the match looks in ("latest" relative to
  what?); the new vocabulary follows `pandas.merge_asof`'s
  backward/forward/nearest convention. `"latest"` still works as a deprecated
  alias and emits a `DeprecationWarning`. The semantics are unchanged: last
  event with `t <= t_ref` (zero-order hold).

## [0.4.0] - 2026-06-25

### Added
- **`remove_channel` -- drop a channel declaration** (the inverse of
  `register_channel` / `register_raw_channel`, which had no counterpart). Removes
  a channel from `channels.yaml` so the dataset stops loading it; the on-disk
  files are kept by default (reversible), and `data=True` / `--purge` also deletes
  the channel's directory. Exposed as `apairo.remove_channel(seq, chan)`, the
  class form `Dataset.remove_channel(...)`, and the CLI `apairo channel remove`
  (root-aware). Removing a **raw** (source) channel or deleting data warns and
  asks for confirmation (`--yes` to skip); a still-referenced channel
  (`timestamps_from` / `sources`) lists its now-dangling dependents.
- **Channel aliases honored by `ProfiledDataset`, not just `RawDataset`** -- a
  profiled dataset now resolves a requested key (alias or real name) to its real
  channel for file discovery while exposing loaders and `sample.data` under the
  public alias, mirroring `AsyncLayoutDataset`. Previously `set_alias` was a no-op
  on profiled datasets (a request by alias raised `KeyError`). This lets one
  pipeline unify channel names across heterogeneous datasets.
- **`ds.calibration.get_tf(source, target)`** -- the static-transform tree is now
  *resolved* in the core, not just exposed. `dataset.calibration` returns a
  `Calibration` (a `dict` subclass, fully backward compatible) whose `get_tf`
  walks the `"<parent>_to_<child>"` edges to return the 4x4 mapping a point from
  `source` into `target` (`p_target = T @ p_source`), composing and inverting as
  needed -- identity when the frames match, `KeyError` when no static path
  connects them. Resolution has exactly one canonical form, so it belongs in the
  core; *applying* the matrix to data (points vs poses vs normals) stays in
  `apairo_transform`. The `frames` docs and schema page are updated accordingly.
- **`ds.calibration` on every dataset, not just `RawDataset`** -- the property now
  reads `<root_dir>/.apairo/calibration.yaml` on any dataset with a root, so the
  synchronous profiled datasets (Rellis/Goose/SemanticKITTI) and the async family
  (`TartanKittiDataset`) all resolve their static tree. Each sensor can sit in its
  own frame regardless of how the dataset is loaded; the async family still merges
  per-sequence tables. Datasets without an on-disk root (cached/concat views)
  return an empty `Calibration`.

### Changed
- **`apairo.dataset.kitti` renamed to `apairo.dataset.async_layout`** -- the
  module held only the abstract `AsyncLayoutDataset` primitive (the class was
  renamed long ago; its module never followed) and no real KITTI dataset. Import
  from the new path: `from apairo.dataset.async_layout import AsyncLayoutDataset`.
- **`RawDataset` bootstraps raw data on load** -- pointing it at a sequence or a
  root that has no `.apairo` now infers the channels (loaders from file
  extensions) and writes the sidecar on the spot, instead of raising and asking
  for an explicit `RawDataset.init()`. `init()` still works and is the way to pin
  loaders or a manifest up front; it is just no longer required to read raw data.
- **`TartanKittiDataset` is now a thin `RawDataset` subclass** (247 -> 40 lines).
  TartanDrive was always "a `RawDataset` whose channels are a fixed set", so the
  class now *is* exactly that: it pins the TartanDrive profile (`available_keys`
  plus a profile-pinned `_bootstrap_config`) and inherits all loading, root,
  synchronization and preprocessing behaviour. Public usage
  (`TartanKittiDataset(seq_or_root, keys=[...])`) is unchanged. The previously
  documented lazy mode (`keys=None` -> no loaders, set `ds.keys` later) is gone:
  `keys=None` now loads every present channel, the same as `RawDataset`.

### Removed
- **Deprecated profile field `torch_dtype`** -- it was the old spelling of
  `cast_dtype`, a historical misnomer (it never touched torch, always resolving to
  a NumPy dtype). Every in-repo profile already uses `cast_dtype`; the back-compat
  shim and its warning are gone. Rename any remaining `torch_dtype` to `cast_dtype`.

### Fixed
- **`timestamps_from` is honored on the whole asynchronous family** -- a channel
  declared with `register_channel(..., timestamps_from=...)` (a derived channel
  with no `timestamps.txt` of its own) now loads through `RawDataset` and every
  `AsyncLayoutDataset`, not just `TartanKittiDataset`. The shared loader used to
  ignore the field and consult only a hardcoded replacement map, so such a channel
  raised when loaded generically. Timestamp resolution (own file -> shared
  `timestamps_from` source -> legacy map) now lives once in
  `AsyncLayoutDataset._collect_timestamps`.

## [0.3.0] - 2026-06-24

### Added
- **Brand identity** -- apairo logo and a badge row (PyPI, Python, CI, license,
  docs) on the README and docs home, plus the mkdocs header/favicon. The README
  logo is a PNG referenced by absolute URL so it renders on PyPI.
- **`apairo check`** -- validates the `.apairo` schema (channels, manifest,
  calibration) and reports issues, exiting non-zero on any (CI-friendly). It is
  profile-aware (same reading as `status`) and consumes `verify_config` /
  `verify_manifest` / `verify_calibration`. (`apairo add` stays deferred to
  post-1.0.)
- **`.apairo` schema frozen & documented as `version: 1`** -- a dedicated docs
  page ("The .apairo Schema") specifies `channels.yaml`, `dataset.yaml` and
  `calibration.yaml` as a stable on-disk contract. `dataset.yaml` now carries
  `version: 1` like the other two. `verify_manifest` and `verify_calibration`
  join `verify_config` (all top-level exports); validation is **tolerant** -- an
  unknown field is reported as a warning and otherwise ignored (forward
  compatible) -- and now also checks `kind`, `transform` structure, and the
  4x4 calibration matrices. The manifest and calibration files stay optional.
- **Channel aliases for `RawDataset`** -- a raw channel can carry an `alias` in
  `.apairo/channels.yaml`, the public name it is loaded and exposed under (e.g.
  the on-disk `ouster_points` directory exposed as `lidar`). The directory keeps
  its real name; `keys=[...]`, `sample.data` and `timestamps` use the alias. This
  brings the profile-free loader the canonical-naming ergonomics profiled
  datasets get from their layout. Set it in Python (`apairo.set_alias(seq, chan,
  alias)` or `register_raw_channel(..., alias=...)`) or from the shell
  (`apairo alias <channel> <alias>`, root-aware); `apairo status` surfaces it.
  A clashing alias (one already in use, or shadowing a real directory name) is
  rejected up front -- it would make the dataset unloadable -- with `--force`
  to reassign an alias from its current holder.
- **Profile-aware `apairo status`** -- a directory initialized with
  `init --as <Class>` is now recognized as that dataset: `status` names the class,
  lists its sequences, and resolves canonical channel names (`lidar`) to their
  real nested directories, instead of the profile-unaware generic reading that
  reported spurious "directory not found" / "unknown loader" issues. The dataset
  class is recorded in `.apairo/dataset.yaml` (the root **manifest**) by
  `ProfiledDataset.init`; `status` dispatches on it.
- **`apairo status -s/--sequence <ID>`** -- per-channel detail for one sequence
  addressed by **id** from the dataset root (`status <root> -s 00000`), instead of
  pointing at the nested sequence directory. For profiled datasets this is the
  only way to inspect a sequence with canonical channel names.
- **`ProfiledDataset.inventory(root)`** -- the path-based, tolerant form of
  `describe()`: structural self-description (identity, sequences, channel->layout
  resolution, splits, calibration) without constructing the dataset.
- **`read_manifest` / `write_manifest`** in `apairo.core.config` -- read/write the
  `.apairo/dataset.yaml` root manifest.

### Deprecated
- **`ProfiledDataset(..., sequence_ids=...)` -> `sequences=`** -- the constructor
  argument that restricts loading to a set of sequences is renamed to `sequences`,
  the symmetric counterpart of `split` (both default to "all"). The old
  `sequence_ids=` keyword is still accepted with a `DeprecationWarning`; the 4th
  positional argument is unchanged. The `sequence_ids` *property* (the list of
  available sequence ids) and `frame_sequence_ids` are unaffected.
- **Profile field `torch_dtype` -> `cast_dtype`** -- the YAML modality field that
  drives the post-load `.astype()` cast is renamed to `cast_dtype`, its honest
  name: it has always resolved to a **NumPy** dtype (`apairo` has no torch
  dependency -- deps are numpy + PyYAML). The old `torch_dtype` spelling is still
  accepted with a `DeprecationWarning`; `cast_dtype` wins if both are present.
  Built-in profiles (rellis, goose, semantic_kitti) updated.

### Changed
- **`transform` and `synchronize` fail loud on misuse.** `transform(fn, "key")`
  (arguments reversed) and `transform("key")` (missing function) now raise
  `TypeError` instead of silently doing nothing, and `synchronize(method={a, b})`
  (a `set` instead of `{a: b}`) raises with a clear hint. These were the two ways
  a terse, correct-looking pipeline could quietly do nothing.
- **`ProfiledDataset.describe()`** now returns a richer **structured** dict
  (identity, `sequences`, `splits`, `calibration`, and per-channel
  `loader`/`dir`/`present`) in addition to the existing `raw`/`preprocess` keys;
  the printed human summary is unchanged. Per-frame facts (counts, shapes) stay
  out of `describe` -- they are recoverable from a loaded dataset (`len(ds)`,
  `ds[i].data[key].shape`).
- **`apairo status` / `init` output is now plain ASCII** (no box-drawing rule,
  em-dash header or arrow glyphs), so it pastes cleanly into ASCII-only contexts.
- **Docs** -- the navigation is grouped into sidebar sections, and the
  coordinate-frame page is renamed "Frames & Calibration" (was "Frames &
  Transforms") to stop colliding with the `.transform()` API page "Transforms".

### Removed
- **`MNTDataset`** — removed from core and moved to the downstream
  `apairo_experiments` repo (`core/datasets/mnt`). It is a specific, internal
  dataset; the canonical public loaders shipped in apairo stay
  `SemanticKittiDataset` / `Rellis3DDataset` / `Goose3DDataset` /
  `TartanKittiDataset` / `RawDataset` ("closed core, open collections"). It keeps
  working as a normal apairo consumer (a `SynchronousDataset` subclass). The
  `mnt` optional-dependency group is replaced by a `zarr` extra, since
  `zarr` is a first-class generic loader (used by `RawDataset`), not MNT-specific.
- **`KittiDataset` alias** — removed (it was a transitional alias for
  `AsyncLayoutDataset` introduced in 0.2.0). `AsyncLayoutDataset` is also no
  longer a top-level `apairo` export: it is now an internal base class, reached
  only by subclassing (`from apairo.dataset.kitti import AsyncLayoutDataset`).
  The public asynchronous loaders are `RawDataset` and `TartanKittiDataset`.
- **`npys_img` loader** — removed (schema cleanup toward the frozen `version: 1`).
  It was a no-op alias of `npys` (same `NPYSLoader`, same `.npy` extension); the
  `img` in the name triggered no image decoding. The one user (TartanDrive's
  `depth_left`) now declares `npys`.
- **`has_timestamps` channel field** — removed from the `.apairo` schema. It was
  written but never read (the loader checks the channel directory on disk), and
  carried no information independent of `kind` / sync-vs-async. The
  `has_timestamps` parameter is gone from `register_raw_channel` (top-level and
  `ConfigurableDataset`).

### Toward 1.0

1.0 is the commitment to a stable public API and `.apairo` format. Remaining:

- [x] **Freeze the public API** — `KittiDataset` alias removed and
  `AsyncLayoutDataset` demoted to an internal base class; no pending renames.
- [x] **Freeze & document the `.apairo` schema** (`channels.yaml`,
  `dataset.yaml`, `calibration.yaml`) as a stable `version: 1` contract.
- [x] **Settle the CLI** — `apairo check` shipped; `apairo add` deferred to
  post-1.0 (`status` already surfaces untracked channels, and they register from
  Python or by re-running `init`). `init` / `status` / `alias` / ecosystem
  dispatch locked.
- [x] **Decide Zarr's scope** — **in**, as an optional first-class loader (its
  own `zarr` extra), like `img`/Pillow. It is generic (`RawDataset` reads zarr),
  not tied to any one dataset.
- [ ] **Soak** on real datasets + the ecosystem roundtrip
  (extractor → apairo → transform / preprocess).

## [0.2.0] - 2026-06-16

First feature release.

### Added
- **`RawDataset`** — profile-free, `channels.yaml`-driven loader; single
  sequence or dataset root (auto-detected); loads `apairo-extractor` output.
  `RawDataset.init` is root-aware (writes per-sequence `channels.yaml` + root
  `dataset.yaml`).
- **`AsyncLayoutDataset`** — the abstract asynchronous-layout base of the async
  family; **`RootSequenceMixin`** factors the shared multi-sequence root
  behaviour (flat index, per-sequence `synchronize` + concat).
- **Loaders / formats** — `npy`, `npys`, `bin`, `img`, and **`zarr`**; the
  channel format is orthogonal to the layout. `DatasetLayout` as the on-disk
  single source of truth.
- **Frames & transforms** (descriptive only — no geometry) — per-channel
  `frame`, dynamic-transform channels (`transform: {parent, child}`), and static
  extrinsics in `.apairo/calibration.yaml` (`read_calibration`,
  `register_static_transform`, `RawDataset.calibration`).
- **`apairo` CLI** — `init` (root-aware) and `status` (per-channel table with
  rate / span relative to start / shape / frame / transform, plus `--json`), and
  ecosystem dispatch `apairo <tool>` via the `apairo.cli_plugins` entry-point
  group (e.g. `apairo extractor`, with no dependency on the tool).
- **Datasets** — `SemanticKittiDataset`, `Goose3DDataset`, `Rellis3DDataset`,
  `TartanKittiDataset`, `MNTDataset`, `StreamDataset`.
- **Composition & views** — `ConcatDataset`, `ZipDataset` / `join`, `filter`,
  `select`, `cache`, access-time `transform` (with multi-channel publishers),
  and `split` / `split_sequences` / `filter_sequences`.
- **Synchronization** — `synchronize()` (asynchronous → synchronous), the
  `Interpolator` interface, and external-clock (fixed-rate / distance) resampling.
- **Preprocessing** — `run_preprocess`, `register_channel`, and the `.apairo`
  integrity check (`verify_config`).
- Documentation site (MkDocs), including Async Datasets, Frames & Transforms,
  and Command Line.

### Changed
- Renamed `KittiDataset` → `AsyncLayoutDataset` (no real KITTI dataset used it).
  `KittiDataset` is kept as a backward-compatible alias — no code change needed.
- Extended the `.apairo` schema (still `version: 1`): per-channel `frame` and
  `transform`, plus a `calibration.yaml` for static extrinsics.

### Fixed
- `ConcatDataset`: key intersection and sub-dataset mutation; hardened the view
  chain (`FilteredView`, `ChannelView`) delegation (`frame_sequence_ids`).
- Removed a dead, matplotlib-based test fixture (image I/O standardizes on Pillow).

[Unreleased]: https://github.com/apairo-robotics/apairo/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/apairo-robotics/apairo/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/apairo-robotics/apairo/compare/v0.6.2...v0.7.0
[0.5.0]: https://github.com/apairo-robotics/apairo/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/apairo-robotics/apairo/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/apairo-robotics/apairo/compare/v0.2.1...v0.3.0
[0.2.0]: https://github.com/apairo-robotics/apairo/compare/v0.1.0...v0.2.0
