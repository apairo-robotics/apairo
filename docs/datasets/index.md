# Datasets

## Supported datasets

| Class | Layout | Modalities | Notes |
|---|---|---|---|
| `SemanticKittiDataset` | synchronous | `lidar`, `labels` | Labels masked to lower 16 bits (strips instance IDs) |
| `Rellis3DDataset` | synchronous | `lidar`, `labels` | Fixed `Rellis-3D/` prefix in directory tree |
| `Goose3DDataset` | synchronous | `lidar`, `labels` | Labels are the class (lower 16 bits); the clock is read from the scan file names |
| `RawDataset` | asynchronous | any channel | Profile-free; channels & loaders from `.apairo/channels.yaml`. Loads `apairo-extractor` output |
| `TartanKittiDataset` | asynchronous | any TartanDrive v2 channel | Fixed channel profile; auto-discovers channels via `.apairo` |

`RawDataset` also reads datasets it has no class for, from a declaration alone.
The declarations of two of them ship with apairo, by name, each with a guide
checked against real sequences:

| Dataset | Declaration | Domain | What the declaration says |
|---|---|---|---|
| [TUM RGB-D](tum-rgbd.md) | `tum_rgbd` | indoor RGB-D SLAM | image clocks in `<sec>.<usec>` filenames, trajectory and accelerometer tables at the sequence root |
| [EuRoC MAV](euroc.md) | `euroc_vicon_room`, `euroc_machine_hall` | drone visual-inertial odometry | image clocks in nanosecond filenames, one `data.csv` per sensor |

### Installing a standard dataset

Every standard dataset ships with apairo. Its extra installs what reading it
needs, and keeps that install line stable if the dataset later moves to a
package of its own:

```bash
pip install apairo[tartan]          # also: rellis, goose, semantic-kitti, tum, euroc
```

A dataset that is not listed can be added with a pull request; see
[Adding a Dataset](adding-a-dataset.md).

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

    The tree of `goose_3d_train.zip` and `goose_3d_val.zip`, unpacked in one
    root:

    ```
    <root>/
      lidar/
        train/
          <scene>/   <scene>__<frame>_<ns>_vls128.bin  ...
        val/
          ...
      labels/
        train/
          <scene>/   <scene>__<frame>_<ns>_goose.label  ...
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
