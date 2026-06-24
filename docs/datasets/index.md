# Datasets

## Supported datasets

| Class | Layout | Modalities | Notes |
|---|---|---|---|
| `SemanticKittiDataset` | synchronous | `lidar`, `labels` | Labels masked to lower 16 bits (strips instance IDs) |
| `Rellis3DDataset` | synchronous | `lidar`, `labels` | Fixed `Rellis-3D/` prefix in directory tree |
| `Goose3DDataset` | synchronous | `lidar`, `labels` | Split directory appears at two levels |
| `RawDataset` | asynchronous | any channel | Profile-free; channels & loaders from `.apairo/channels.yaml`. Loads `apairo-extractor` output |
| `TartanKittiDataset` | asynchronous | any TartanDrive v2 channel | Fixed channel profile; auto-discovers channels via `.apairo` |

---

## Synchronous datasets

All three synchronous datasets share the same interface because they all extend [`ProfiledDataset`][apairo.core.profiled_dataset.ProfiledDataset].

```python
import apairo

# SemanticKITTI
ds = apairo.SemanticKittiDataset("/data/semantic_kitti", keys=["lidar", "labels"])

# GOOSE
ds = apairo.Goose3DDataset("/data/goose", keys=["lidar", "labels"], split="train")

# Rellis-3D
ds = apairo.Rellis3DDataset("/data/rellis", keys=["lidar"])
```

### Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `root_dir` | `str \| Path` | -- | Dataset root directory |
| `keys` | `list[str] \| None` | all non-optional keys | Modalities to load |
| `split` | `str \| None` | `None` | Restrict to a named split (e.g. `"train"`, `"val"`, `"test"`) |
| `sequences` | `list[str] \| None` | all | Restrict to these sequence ids -- the symmetric counterpart of `split` |

### Expected directory layouts

=== "SemanticKITTI"

    ```
    <root>/
      sequences/
        00/
          velodyne/   000000.bin  000001.bin  ...
          labels/     000000.label  000001.label  ...
        01/
          ...
    ```

=== "GOOSE"

    ```
    <root>/
      lidar/
        train/
          seq_001/   000000.bin  000001.bin  ...
          seq_002/   ...
        val/
          ...
      labels/
        train/
          seq_001/   000000.label  ...
        val/
          ...
    ```

=== "Rellis-3D"

    ```
    <root>/
      Rellis-3D/
        00000/
          os1_cloud_node_kitti_bin/    000000.bin  ...
          os1_cloud_node_kitti_label/  000000.label  ...
        00001/
          ...
    ```

---

## How synchronous datasets work

Each class declares a YAML structural profile that describes the directory layout, file extensions, dtypes, and any type transformations needed. apairo reads the profile at import time and handles discovery, loading, and casting automatically.

See [YAML Profiles](yaml-profiles.md) for the full specification, and [Adding a Dataset](adding-a-dataset.md) to create your own.
