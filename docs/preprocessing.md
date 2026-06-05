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

    def process(self, sample: Sample) -> np.ndarray:
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

---

## SequencePreprocessor

`SequencePreprocessor` receives an iterator over all frames and returns a single array for the whole sequence. Use it for algorithms that need global context: ICP registration, trajectory smoothing, global statistics.

```python
class GICPPoses(SequencePreprocessor):
    output_key    = "gicp_poses"
    output_loader = "npy"           # single stacked .npy file
    input_keys    = ["velodyne_0"]
    sources       = ["velodyne_0"]  # has its own timestamps.txt in output

    def process(self, frames) -> np.ndarray:
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
    has_timestamps: false
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
