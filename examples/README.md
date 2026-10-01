# Examples

Runnable pipelines that show apairo on real dataset layouts. Each script points at
a dataset root through an environment variable (or a `--root` flag / an in-file
constant) and falls back to a conventional path, so set the variable for your
machine before running:

```bash
export APAIRO_RELLIS_ROOT=/data/RELLIS
export APAIRO_TARTAN_SEQ=/data/tartan/2024-01-01_forest
python examples/tartan_kitti_basic.py
```

The RELLIS examples consume a derived `trav_gt` channel — run
`rellis_traversability.py` first to build it. `join_persisted_prior.py` runs out of
the box against a bundled test asset, no dataset needed.

| Script | Demonstrates | Dataset / env |
|---|---|---|
| `tum_rgbd_associate.py` | Read a TUM RGB-D sequence as downloaded, from a declaration only, and associate depth, pose and accelerometer to each colour frame within 20 ms — see the [guide](../docs/datasets/tum-rgbd.md) | TUM RGB-D — `APAIRO_TUM_SEQ` |
| `euroc_synchronize.py` | Read a EuRoC MAV sequence (ASL format) from a declaration only and put the IMU and the ground-truth state on the camera clock — see the [guide](../docs/datasets/euroc.md) | EuRoC MAV — `APAIRO_EUROC_SEQ` (the `mav0/` directory) |
| `tartan_kitti_basic.py` | Load a TartanDrive v2 sequence and iterate its async event timeline | TartanDrive — `APAIRO_TARTAN_SEQ` |
| `tartan_synchronize.py` | `synchronize()` an async sequence onto a reference clock; inspect per-channel staleness with `time_offsets` | TartanDrive — `APAIRO_TARTAN_SEQ` |
| `tartan_kitti_preprocess.py` | Register and run frame- and sequence-level preprocessors, persisted to `.apairo` | TartanDrive — `APAIRO_TARTAN_SEQ` |
| `tartan_frame_transform.py` | Resolve a static extrinsic with `calibration.get_tf` / `register_static_transform` and apply it to move lidar into `base_link` | TartanDrive — `APAIRO_TARTAN_SEQ` |
| `rellis_traversability.py` | Derive the `trav_gt` channel from RELLIS semantic labels — the prerequisite for the other RELLIS examples | RELLIS-3D — `APAIRO_RELLIS_ROOT` |
| `rellis_camera_sync.py` | Load the Rellis camera + sparse image-labels through the async family, keying both off the timestamp in each filename | RELLIS-3D (camera) — `APAIRO_RELLIS_ROOT` |
| `training_pipeline_rellis.py` | End-to-end traversability pipeline: preprocess, built-in splits, and the `cache()` boundary between deterministic and stochastic ops | RELLIS-3D — `APAIRO_RELLIS_ROOT` |
| `sequence_kfold.py` | Sequence-level k-fold with `filter_sequences`; persist filtered indices to skip the sweep on later runs | RELLIS-3D — `APAIRO_RELLIS_ROOT` |
| `join_cached_prior.py` | Freeze an expensive deterministic prior in RAM with `select().cache()`, then `join` it onto a live stochastically-augmented dataset | RELLIS-3D — `APAIRO_RELLIS_ROOT` |
| `join_persisted_prior.py` | Persist a stateful sequential derived channel to disk with `ChannelWriter` and `join` it back — reload free on the second run | bundled test asset (override with `APAIRO_ROOT` / `APAIRO_CACHE_ROOT`) |
| `cross_dataset_concat.py` | Train across RELLIS + GOOSE + SemanticKITTI: per-dataset normalization, key renaming, and `repeat()` to rebalance before `concat` | RELLIS + GOOSE + SemanticKITTI (roots set in file) |
| `goose_traversability.py` | A `FramePreprocessor` mapping GOOSE semantic labels to a binary traversability mask via a YAML config | GOOSE-3D — `--root` / `--config` |
| `transforms_basic.py` | At-access transforms: per-channel, sample-level mask, and a published `keep=False` intermediate | GOOSE-3D — `--root` |
