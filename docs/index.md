<p align="center">
  <img src="resources/apairo_logo_full.svg" alt="apairo" width="480">
</p>

**Unified robotics dataset loader for time-series and annotated sensor data.**

apairo handles the two fundamental layouts found in robotics datasets:

- **Synchronous** -- every index `i` returns a complete co-captured frame (semantic segmentation datasets)
- **Asynchronous** -- multiple sensors firing at different rates, interleaved into a single timestamp-ordered timeline (KITTI-style multi-modal recordings)

---

## At a glance

```python
import apairo

# Synchronous: SemanticKITTI
ds = apairo.SemanticKittiDataset("/data/semantic_kitti", keys=["lidar", "labels"])
sample = ds[0]
# sample.data["lidar"]   -> torch.Tensor (N, 4)  float32
# sample.data["labels"]  -> torch.Tensor (N,)    int64

# Asynchronous: TartanDrive
ds = apairo.TartanKittiDataset("/data/tartan/2024-01-01_forest")
sample = ds[0]
# sample.data        -> {"velodyne_0": tensor}
# sample.timestamp   -> float
```

---

## Supported datasets

| Class | Layout | Modalities |
|---|---|---|
| `SemanticKittiDataset` | synchronous | lidar, labels |
| `Rellis3DDataset` | synchronous | lidar, labels |
| `Goose3DDataset` | synchronous | lidar, labels |
| `TartanDataset` | synchronous | any (`.pt` format) |
| `RawDataset` | asynchronous | any channels (from `.apairo/channels.yaml`) |
| `TartanKittiDataset` | asynchronous | any TartanDrive v2 channel |

---

## Key features

- **YAML-driven dataset profiles** -- adding a new synchronous dataset requires one `.yaml` file and two lines of Python
- **Derived key loading** -- preprocessed outputs live alongside raw data, registered in a `.apairo` sidecar and loaded transparently
- **Preprocessing framework** -- `FramePreprocessor` and `SequencePreprocessor` run pipelines and persist results automatically ([`apairo_preprocess`](https://github.com/apairo/apairo_preprocess) for ready-made preprocessors)
- **At-access transforms** -- `dataset.transform(key, fn)` and `Compose` apply callables at read time without writing to disk ([`apairo_transform`](https://github.com/apairo/apairo_transform) for ready-made transforms)
- **PyTorch integration** -- `TorchConcatDataset` and `TorchKittiDataset` wrap any dataset for use with `DataLoader`
- **Sequence-level splits** -- `split_sequences()` avoids temporal leakage across train/val/test

---

[Get started ->](quickstart.md){ .md-button .md-button--primary }
[Installation ->](installation.md){ .md-button }
