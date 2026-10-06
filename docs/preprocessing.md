# Preprocessing

apairo provides a framework for running and persisting preprocessing pipelines alongside datasets.

> **Companion library:** [`apairo_preprocess`](https://github.com/apairo/apairo_preprocess) ships ready-made preprocessors (ICP registration, normal estimation, ground removal, …) built on top of this API. Results are stored following the dataset's layout conventions and registered automatically in a `.apairo` sidecar.

---

## FramePreprocessor

`FramePreprocessor` runs your function once per frame. Use it for per-scan operations: label inference, feature extraction, normal estimation, etc.

```python
import numpy as np
from apairo import FramePreprocessor, Goose3DDataset
from apairo.core.sample import Sample


class TraversabilityLabel(FramePreprocessor):
    output_key      = "trav_label"   # output subdirectory name
    output_loader   = "npys"         # one .npy file per frame
    input_keys      = ["labels"]     # channels needed as input
    timestamps_from = "labels"       # inherit timestamps, no timestamps.txt written
    sources         = ["labels"]     # provenance metadata in .apairo

    def __call__(self, sample: Sample) -> np.ndarray:
        labels = sample.data["labels"]          # np.ndarray (N,)
        traversable_ids = {1, 2, 5, 9}
        mask = np.zeros(len(labels), dtype=bool)
        for i in traversable_ids:
            mask |= labels == i
        return mask.astype(np.uint8)            # (N,)  uint8


Goose3DDataset.run_preprocess(TraversabilityLabel(), "/data/goose/seq_001")
```

After running, the output is available as a loadable key:

```python
ds = Goose3DDataset("/data/goose/seq_001", keys=["lidar", "trav_label"])
sample = ds[0]
print(sample.data["trav_label"].shape)   # torch.Size([N])
```

### Preview before materializing

A `FramePreprocessor` is a callable on a `Sample` — the same protocol as a
transform. Pass it to `transform()` to run it lazily: the result is published
under its `output_key` at access time, nothing is written to disk. Iterate on
the implementation, visualize a few frames, then materialize the same object:

```python
p = TraversabilityLabel()
preview = ds.transform(p)             # lazy, nothing written
preview[42].data["trav_label"]        # computed on access — inspect, plot
ds.run_preprocess(p)                  # once satisfied, persist + register
```

(A `SequencePreprocessor` cannot run lazily — it needs the full sequence at
once, which is exactly why materialization exists.)

---

## SequencePreprocessor

`SequencePreprocessor` receives an iterator over all frames and returns a single array for the whole sequence. Use it for algorithms that need global context: ICP registration, trajectory smoothing, global statistics.

```python
class GICPPoses(SequencePreprocessor):
    output_key    = "gicp_poses"
    output_loader = "npy"           # single stacked .npy file
    input_keys    = ["velodyne_0"]
    sources       = ["velodyne_0"]  # has its own timestamps.txt in output

    def __call__(self, frames) -> np.ndarray:
        poses = []
        for sample in frames:
            pts = sample.data["velodyne_0"]
            poses.append(register_icp(pts))     # your function
        return np.stack(poses)   # (N, 4, 4)


TartanKittiDataset.run_preprocess(GICPPoses(), "/data/tartan/seq_001")
```

---

## Class attributes reference

| Attribute | Type | Description |
|---|---|---|
| `output_key` | `str` | Subdirectory name for the output channel |
| `output_keys` | `list[str]` | Multi-output alternative to `output_key` (exclusive). `__call__` returns a `dict` with exactly these keys; one derived channel is written and registered per key, all sharing `output_loader` and provenance. |
| `output_loader` | `str \| None` | The format to write the output in -- any format that writes (see [the output format](#the-output-format)). Leave it unset unless the output's nature calls for one. |
| `input_keys` | `list[str]` | Dataset channels required as input |
| `timestamps_from` | `str \| None` | If set to a channel name, the output inherits that channel's timestamps and no `timestamps.txt` is written. If `None`, timestamps are written from the input sample timestamps. |
| `sources` | `list[str] \| None` | Provenance recorded in `.apairo` for reference. |

---

## Multi-output preprocessors

One expensive pass sometimes produces several channels at once -- a voxel
structure emits `cell_coords` and `cell_inv` together, a ray accumulator emits
hits, throughs and variance from the same traversal. Declare `output_keys`
instead of `output_key` and return a `dict` with exactly those keys:

```python
class VoxStructure(SequencePreprocessor):
    output_keys   = ["cell_coords", "cell_inv"]
    output_loader = "npys"
    input_keys    = ["ouster_points", "pose"]

    def __call__(self, frames):
        coords, inv = voxelize([s.data["ouster_points"] for s in frames])
        return {"cell_coords": coords, "cell_inv": inv}
```

The runner writes one derived channel per key (same `output_loader`) and
registers all of them in `.apairo` with shared `timestamps_from`/`sources`
provenance -- each channel stays individually selectable. The lazy preview
(`ds.transform(prep)`, `FramePreprocessor` only) publishes every key of the
dict; `output=` cannot rename a multi-output preprocessor.

---

## Several inputs

A preprocessor may declare several `input_keys`. On a synchronous dataset every
sample already holds all of them. On an asynchronous dataset each timeline index
holds a single channel, so the runner looks at the inputs' clocks:

- **Inputs on one clock are grouped.** When the inputs have identical timestamps
  -- typically channels derived from the same sensor, such as a voxelised cloud
  and the ground labels computed from it -- row `i` of each is one moment. The
  runner zips them row for row: the preprocessor receives one sample per row,
  holding every input, and the output is numbered by row and stamped by that
  clock.

    ```python
    RawDataset.run_preprocess(VoxelisePointCloud(lidar_key="velodyne_0"), seq)
    RawDataset.run_preprocess(GroundSegmentationRANSAC(), seq)       # voxelised -> ground_ransac
    RawDataset.run_preprocess(                                       # two inputs, one clock
        GroundHeightFromLabels(ground_key="ground_ransac"), seq
    )
    ```

- **Inputs on different clocks are refused**, by name and with each channel's
  frame count, before anything is written. Pairing a 10 Hz lidar with a 50 Hz
  pose is a synchronisation, with a method and a tolerance to choose; the
  runner does not guess one. Running a preprocess over a `synchronize()` view is
  not supported yet. Until it is, write the channel yourself from the view with
  [`ChannelWriter`](#channelwriter-channels-produced-outside-apairo).

"Identical" means the same timestamps, not merely the same number of frames: two
channels of equal length on shifted clocks are refused too.

---

## The output format

apairo does not impose a storage format. Each output channel is written in the
first format of this list that applies:

1. the format the run asks for: `run_preprocess(prep, root, output_format="zarr")`;
2. the preprocessor's `output_loader`, when it declares one;
3. the format of its input channel (its `timestamps_from` channel, else its
   first input), when that format can hold the output. An image mask stays a
   PNG image, and a crop of `.xyz` clouds stays `.xyz` clouds;
4. otherwise `npys`, one `.npy` per frame, for a `FramePreprocessor`, or
   `npy`, one stacked array, for a `SequencePreprocessor`.

The choice is made on the first output, before anything is written, and each
output channel is recorded in `.apairo` with its format. In cases 1 and 2,
apairo refuses a format that cannot hold the output and names the format. A
read-only format is refused with the list of formats that write.

| Format | Writes | Holds |
| --- | --- | --- |
| `npys` | one `.npy` per frame | any numeric array |
| `npy` | one stacked `.npy` | any numeric array, one row per frame |
| `img` | one `.png` per frame | `uint8` images (grey, RGB, RGBA), `uint16` depth maps |
| `bin` | one `.bin` per frame | `(N, 4)` `float32` points |
| `zarr` | the channel directory is the store | any numeric array, one row per frame |
| `csv` | one table, its clock in `timestamps.txt` | `(N, k)` `float64` rows |
| `pcd` | read-only for now | |

A [format plugin](datasets/format-plugins.md) that writes is a target like
the others. What a format writes, it reads back unchanged: the conformance
check `check_format` verifies it.

A stacked format (`npy`, `zarr`, `csv`) can also take a `FramePreprocessor`'s
output: the frames of each sequence are stacked once the run is over, which
needs one shape per frame.

### A format conversion is a preprocess

Copying a channel into another format is just another derived channel. It
is written by the runner, stamped with the source's clock, and recorded with
its `sources` and recipe:

```python
from apairo.preprocess import Convert

RawDataset.run_preprocess(Convert("velodyne_0", to="zarr"), seq)
# -> seq/velodyne_0_zarr/, a zarr channel with sources [velodyne_0]
RawDataset.run_preprocess(Convert("imu", to="npy", output_key="imu_array"), seq)
```

---

## Overwrite protection

By default, `run_preprocess` raises `FileExistsError` if an output channel's directory already holds data (every declared key is checked). Pass `overwrite=True` to recompute:

```python
Goose3DDataset.run_preprocess(preprocessor, "/data/goose", overwrite=True)
```

---

## The `.apairo` directory

After a successful run, `run_preprocess` writes or updates `.apairo/channels.yaml` at the dataset root:

```
dataset_root/
└── .apairo/
    └── channels.yaml
```

```yaml
version: 1
channels:
  trav_label:
    kind: preprocess
    loader: npys
    sources: [labels]
```

This file is read automatically on the next dataset load -- no code change needed to use the new key. The `.apairo/` directory can be deleted entirely to reset a dataset to its raw state without touching any data.

---

## Output file placement

Output files are placed using `dataset.derived_path(idx, output_key, ext)`. For `ProfiledDataset` subclasses, this replaces the modality component in the source file path:

| Source | Derived |
|---|---|
| `lidar/train/seq_a/000000.bin` | `trav_label/train/seq_a/000000.npy` |
| `sequences/00/velodyne/000000.bin` | `sequences/00/trav_label/000000.npy` |
| `Rellis-3D/00000/os1_cloud_node_kitti_bin/000000.bin` | `Rellis-3D/00000/trav_label/000000.npy` |

The placement is consistent with each dataset's native structure, so derived files sit naturally alongside raw data.

---

## ChannelWriter -- channels produced *outside* apairo

`run_preprocess` is the path for a **deterministic** derived channel: a
callable apairo runs per frame. Some channels don't fit that
shape -- they are produced by an **external tool** (a labeling/annotation app, a
one-off script) that already *holds* the data, often for only a **few frames**
(e.g. ground-truth labels used only at evaluation). For those, use
`ChannelWriter`.

It owns the three things that make a channel loadable, so an external tool never
re-implements -- and drifts from -- the on-disk format:

1. the **frame-naming policy** the loader reads back (a frame stem must not
   contain `_`, which is reserved for suffixed sub-channel variants like
   `000000_intensity.npy`);
2. a `timestamps.txt` kept in the frame order the loader sorts to;
3. **registration** in `.apairo/channels.yaml` on `close()`.

```python
import apairo

# A labeling tool wrote per-point ground truth for one lidar frame (001813).
with apairo.ChannelWriter(seq_dir, "ground_truth", loader="npys",
                          timestamps_from="ouster_points",
                          sources=["ouster_points"]) as w:
    w.add(labels, stem="001813", timestamp=t)   # -> seq/ground_truth/001813.npy
# channels.yaml now declares ground_truth (kind: preprocess); it loads like any
# other channel and synchronizes onto its source by timestamp.
```

- **Per-frame loaders only** (`npys`, `bin`); the stacked `npy` and `img`/`zarr`
  are out of scope. `npys` preserves any dtype (use it for integer labels).
- The `timestamp` is the source frame's timestamp -- alignment in an async
  dataset is by `timestamps.txt`, not by filename, so the stem is free (it just
  must not contain `_`). Mirroring the source stem (`001813`) keeps it legible.
- Re-opening a writer on an existing channel **resumes** it: previously written
  frames are picked up, so you can annotate incrementally across runs.
- A channel may hold a single frame; it loads and synchronizes the same way.

`ChannelWriter` writes the format; *which* frames are train vs eval stays a view
concern (`filter` / a frozen index file), never baked into the layout.
