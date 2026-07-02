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
| `output_loader` | `str` | Storage format: `"npys"` (one file/frame), `"npy"` (stacked), `"bin"` (raw binary/frame), `"pt"` |
| `input_keys` | `list[str]` | Dataset channels required as input |
| `timestamps_from` | `str \| None` | If set to a channel name, the output inherits that channel's timestamps and no `timestamps.txt` is written. If `None`, timestamps are written from the input sample timestamps. |
| `sources` | `list[str] \| None` | Provenance recorded in `.apairo` for reference. |

---

## Overwrite protection

By default, `run_preprocess` raises `FileExistsError` if the first output file already exists. Pass `overwrite=True` to recompute:

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
