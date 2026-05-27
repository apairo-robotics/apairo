# apairo

Unified robotics dataset loader for time-series and annotated sensor data.

Handles the two fundamental layouts found in robotics datasets:

- **Synchronous** — every index `i` returns a complete co-captured frame (semantic segmentation datasets, `.pt` files)
- **Asynchronous** — multiple sensors firing at different rates, interleaved into a single timestamp-ordered timeline (KITTI-style multi-modal recordings)

---

## Installation

```bash
pip install -e .
```

And soon 
```bash
pip install apairo
```

Requires Python ≥ 3.11.  Optional visualization extras: `pip install apairo[viz]`.

---

## Quickstart

```python
import apairo

# Synchronous: SemanticKITTI — index i returns one complete frame
ds = apairo.SemanticKittiDataset("/data/semantic_kitti", keys=["lidar", "labels"])
sample = ds[0]
# sample.data["lidar"]   → torch.Tensor (N, 4)  float32   [x, y, z, intensity]
# sample.data["labels"]  → torch.Tensor (N,)    int64
# sample.timestamp       → None

# Asynchronous: TartanDrive — index i returns one event from the merged timeline
ds = apairo.TartanKittiDataset("/data/tartan/2024-01-01_forest")
sample = ds[0]
# sample.data        → {"velodyne_0": tensor}   (one modality per event)
# sample.timestamp   → float
```

See [`examples/`](examples/) for complete usage patterns.

---

## Supported datasets

| Class | Layout | Modalities |
|---|---|---|
| `SemanticKittiDataset` | synchronous | lidar, labels |
| `Rellis3DDataset` | synchronous | lidar, labels |
| `Goose3DDataset` | synchronous | lidar, labels |
| `TartanDataset` | synchronous | any (`.pt` format) |
| `TartanKittiDataset` | asynchronous | any TartanDrive v2 channel |
| `KittiDataset` | asynchronous | any KITTI-layout channel |

---

## Preprocessing

apairo provides a framework for running and persisting preprocessing pipelines alongside datasets.
Results are stored in the sequence directory following the dataset's layout conventions and registered automatically.

```python
from apairo.preprocess import FramePreprocessor
from apairo.dataset import TartanKittiDataset

class TravLabel(FramePreprocessor):
    output_key      = "trav_label"
    output_loader   = "npys"
    input_keys      = ["velodyne_0"]
    timestamps_from = "velodyne_0"

    def process(self, sample):
        return my_model(sample.data["velodyne_0"])  # → np.ndarray

TartanKittiDataset.run_preprocess(TravLabel(), "/data/tartan/2024-01-01_forest")
# writes  trav_label/000000.npy, 000001.npy, ...
# updates .apairo config automatically
```

The sequence can then be loaded including the new channel:

```python
ds = TartanKittiDataset(
    "/data/tartan/2024-01-01_forest",
    keys=["velodyne_0", "trav_label"],
)
```

---

## Combining datasets

```python
sequences = [
    apairo.SemanticKittiDataset(f"/data/kitti/seq_{i:02d}", keys=["lidar", "labels"])
    for i in range(4)
]

# Concatenate
combined = apairo.ConcatDataset(sequences)

# Sequence-level train / val / test split (avoids temporal leakage)
train, val, test = apairo.split_sequences(sequences, ratios=(0.8, 0.1, 0.1))
```

---

## PyTorch integration

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    apairo.TorchConcatDataset(sequences),
    batch_size=8,
    shuffle=True,
    num_workers=4,
)
```

---

## Extending apairo

Implement `SynchronousDataset` for co-captured frames or `KittiDataset` +
`ConfigurableDataset` for asynchronous extensible datasets.
See [`apairo/dataset/goose/`](apairo/dataset/goose/) and
[`apairo/dataset/tartan_kitti/`](apairo/dataset/tartan_kitti/) as references.

---

## License

MIT
