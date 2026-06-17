# Changelog

All notable changes to apairo are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/) — the public API and the on-disk
`.apairo` format stabilize at **1.0**.

## [Unreleased]

### Removed
- **`KittiDataset` alias** — removed (it was a transitional alias for
  `AsyncLayoutDataset` introduced in 0.2.0). `AsyncLayoutDataset` is also no
  longer a top-level `apairo` export: it is now an internal base class, reached
  only by subclassing (`from apairo.dataset.kitti import AsyncLayoutDataset`).
  The public asynchronous loaders are `RawDataset` and `TartanKittiDataset`.

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
