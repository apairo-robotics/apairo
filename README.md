# apairo

Unified Python loader for robotics sensor datasets -- synchronous (SemanticKITTI, GOOSE, Rellis-3D) and asynchronous (TartanDrive, KITTI) layouts with built-in preprocessing pipelines.

All data is returned as `numpy.ndarray`. Convert to the framework of your choice.

---

## Installation

```bash
pip install -e .
```

And soon
```bash
pip install apairo
```

Optional extras:

```bash
pip install apairo[torch]    # PyTorch support (.pt files)
pip install apairo[vision]   # Image loading (Pillow)
```

Requires Python ≥ 3.11.

---

## Quickstart

```python
import apairo

# Synchronous: SemanticKITTI -- index i returns one complete frame
ds = apairo.SemanticKittiDataset("/data/semantic_kitti", keys=["lidar", "labels"])
sample = ds[0]
# sample.data["lidar"]   -> np.ndarray (N, 4)  float32  [x, y, z, intensity]
# sample.data["labels"]  -> np.ndarray (N,)    int64
# sample.timestamp       -> None

# Asynchronous: TartanDrive -- index i returns one event from the merged timeline
ds = apairo.TartanKittiDataset("/data/tartan/2024-01-01_forest")
sample = ds[0]
# sample.data        -> {"velodyne_0": np.ndarray}   (one modality per event)
# sample.timestamp   -> float
```

### Framework conversion

```python
# PyTorch
import torch
lidar = torch.from_numpy(sample.data["lidar"])

# JAX
import jax.numpy as jnp
lidar = jnp.array(sample.data["lidar"])

# TensorFlow
import tensorflow as tf
lidar = tf.constant(sample.data["lidar"])
```

See [`examples/`](examples/) for complete usage patterns.

---

## Supported datasets

| Class | Layout | Modalities |
|---|---|---|
| `SemanticKittiDataset` | synchronous | lidar, labels |
| `Rellis3DDataset` | synchronous | lidar, labels |
| `Goose3DDataset` | synchronous | lidar, labels |
| `TartanKittiDataset` | asynchronous | any TartanDrive v2 channel |
| `KittiDataset` | asynchronous | any KITTI-layout channel |

---

## Preprocessing

Persist computed channels alongside raw data with `FramePreprocessor` or `SequencePreprocessor`.
See [`apairo_preprocess`](https://github.com/apairo/apairo_preprocess) for a collection of ready-made preprocessors.

```python
from apairo.preprocess import FramePreprocessor
from apairo.dataset import Goose3DDataset
import numpy as np

class TravLabel(FramePreprocessor):
    output_key      = "trav_label"
    output_loader   = "npys"
    input_keys      = ["labels"]
    timestamps_from = "labels"

    def process(self, sample) -> np.ndarray:
        return (sample.data["labels"] < 10).astype(np.uint8)

Goose3DDataset.run_preprocess(TravLabel(), "/data/goose")
# writes  trav_label/train/seq/000000.npy, ...
# updates .apairo config automatically
```

---

## Transforms

Apply callables to channel data at access time -- no disk writes.
See [`apairo_transform`](https://github.com/apairo/apairo_transform) for a collection of ready-made transforms.

```python
from apairo import Compose

ds = apairo.Goose3DDataset("/data/goose", keys=["lidar", "labels"])
ds.transform("lidar", Compose([RangeFilter(max_range=50), ZNorm()]))

sample = ds[0]   # transform applied transparently
```

---

## Combining datasets

```python
# One instance loads all sequences under the root automatically
ds = apairo.SemanticKittiDataset("/data/kitti/dataset", keys=["lidar", "labels"])

# Datasets with a split layer support train/val/test filtering
ds_train = apairo.Goose3DDataset("/data/goose/GOOSE_3D", keys=["lidar", "labels"], split="train")
ds_val   = apairo.Goose3DDataset("/data/goose/GOOSE_3D", keys=["lidar", "labels"], split="val")

# Combine multiple independent datasets (e.g. different data sources)
combined = apairo.ConcatDataset([ds_train, ds_val])
```

---

## PyTorch DataLoader

```python
from torch.utils.data import DataLoader

loader = DataLoader(apairo.ConcatDataset(sequences), batch_size=8, shuffle=True)

for batch in loader:
    lidar = torch.from_numpy(batch["lidar"])   # (8, N, 4)
```

---

## Extending apairo

Add a new synchronous dataset with a YAML profile and a 2-line subclass.
See [documentation](https://apairo.readthedocs.io) for the full guide.

---

## License

MIT
