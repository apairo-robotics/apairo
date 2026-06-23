# Changelog

All notable changes to apairo are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/) — the public API and the on-disk
`.apairo` format stabilize at **1.0**.

## [Unreleased]

### Added
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
- **Profile field `torch_dtype` -> `cast_dtype`** -- the YAML modality field that
  drives the post-load `.astype()` cast is renamed to `cast_dtype`, its honest
  name: it has always resolved to a **NumPy** dtype (`apairo` has no torch
  dependency -- deps are numpy + PyYAML). The old `torch_dtype` spelling is still
  accepted with a `DeprecationWarning`; `cast_dtype` wins if both are present.
  Built-in profiles (rellis, goose, semantic_kitti) updated.

### Changed
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
- [ ] **Freeze & document the `.apairo` schema** (`channels.yaml`,
  `dataset.yaml`, `calibration.yaml`) as a stable `version: 1` contract.
- [ ] **Settle the CLI** — ship or explicitly defer `apairo add` / `apairo check`;
  lock `init` / `status` / ecosystem dispatch.
- [ ] **Decide Zarr's scope** (in or out of 1.0).
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

[Unreleased]: https://github.com/apairo-robotics/apairo/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/apairo-robotics/apairo/compare/v0.1.0...v0.2.0
