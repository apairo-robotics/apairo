# Changelog

All notable changes to apairo are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/) — the public API and the on-disk
`.apairo` format stabilize at **1.0**.

## [Unreleased]

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

[Unreleased]: https://github.com/apairo-robotics/apairo/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/apairo-robotics/apairo/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/apairo-robotics/apairo/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/apairo-robotics/apairo/compare/v0.2.1...v0.3.0
[0.2.0]: https://github.com/apairo-robotics/apairo/compare/v0.1.0...v0.2.0
