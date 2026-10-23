# Concepts

## Synchronous vs asynchronous

apairo distinguishes two fundamental ways that sensor data is organised on disk.

### Synchronous

In synchronous datasets, all modalities at frame `i` were captured at the same instant. Semantic segmentation datasets (SemanticKITTI, GOOSE, Rellis-3D) follow this layout: for every scan there is exactly one `.bin` point cloud file and one `.label` annotation file.

```
sequences/
  00/
    velodyne/  000000.bin  000001.bin  ...
    labels/    000000.label 000001.label ...
```

`ds[i]` returns a `Sample` whose `data` dict contains all requested keys. Synchrony is structural (`ds.is_synchronous`) and says nothing about a clock: a synchronous dataset may still expose a shared per-frame clock through `ds.timestamps`, in which case `sample.timestamp` is that frame's tick. Without one it stays `None`. Random access and standard PyTorch `DataLoader` shuffling work out of the box.

### Asynchronous

In asynchronous datasets, stored one directory per channel, each sensor has its own subdirectory with its own file sequence and a `timestamps.txt`. The sensors fire at different rates.

```
velodyne_0/   000000.bin   000001.bin   ...   timestamps.txt
image_left/   000000.png   000001.png   ...   timestamps.txt
imu/          000000.npy   000001.npy   ...   timestamps.txt
```

apairo merges all channels into a single timestamp-ordered timeline. `ds[i]` returns one event -- the scan or image or IMU reading at position `i` in the global timeline -- with its `timestamp` field set. Exactly one key is populated in `sample.data` per event.

### Bridging the two: `synchronize()`

An asynchronous dataset can be resampled onto a reference clock with `ds.synchronize()`. The result is a synchronous view -- complete multi-channel samples, random access, full chaining API -- making the two layouts interchangeable downstream. See [Synchronization](synchronization.md).

---

## ProfiledDataset and YAML profiles

All synchronous dataset classes in apairo are backed by a YAML *structural profile* that describes the folder layout and data types. The profile replaces what would otherwise be boilerplate Python code in every dataset subclass.

A 2-line Python class:

```python
class Goose3DDataset(ProfiledDataset):
    _profile = "goose.yaml"
```

…combined with a YAML profile handles discovery, loading, splitting, and type casting automatically. See [YAML Profiles](datasets/yaml-profiles.md) for the full specification.

---

## The `.apairo` sidecar

When you run a preprocessing pipeline on a dataset, the output files are written alongside the original data and their location is recorded in a `.apairo` YAML sidecar file at the dataset root (or sequence directory).

```yaml
# .apairo
version: 1
channels:
  trav_label:
    kind: preprocess
    loader: npys
    sources: [labels]
```

On the next load, apairo reads `.apairo` to discover where derived keys live and which loader to use -- no code change needed:

```python
ds = Goose3DDataset("/data/goose", keys=["lidar", "trav_label"])
```

The sidecar is created and updated automatically by `run_preprocess`. You can also write it manually for ad-hoc derived data.

---

## Which storage to choose

What apairo standardizes is the `.apairo` contract, not a storage: the sidecar describes the data where it lies, each channel with its format and its clock. A directory per channel, with one file per frame and a `timestamps.txt`, is the default because numpy alone reads it and `ls` shows it. It is not KITTI: KITTI tools do not read it, and KITTI's own datasets load through their profiles.

| Storage | Formats | Choose it for |
|---|---|---|
| One file per frame | `npys`, `bin`, `img`, `pcd` | Large frames read one at a time, such as scans and images; inspecting or replacing one frame with ordinary tools. |
| One stacked file per channel | `npy`, `csv` | Small frames at a high rate, such as an IMU, poses or forces: at 200 Hz, one file per frame makes 12,000 files a minute. |
| A Zarr store per channel | `zarr` (`pip install apairo[zarr]`) | Chunked and compressed arrays; many frames on a network filesystem, where opening a file costs more than reading it. |

Moving a channel from one storage to another is a preprocess. The new channel is registered in `.apairo` beside its source, with the source's clock:

```python
from apairo import Convert, RawDataset

RawDataset.run_preprocess(Convert("imu", to="zarr"), seq)   # -> seq/imu_zarr/
```

One container holding a whole recording, such as an HDF5 file or a Zarr group of several channels, is not read yet.

---

## Key resolution order

When you request `keys=["lidar", "trav_label"]`:

1. `lidar` is found in the YAML profile -> loaded via the binary path
2. `trav_label` is not in the profile -> looked up in `.apairo` -> loaded via `DERIVED_LOADERS`

At least one native key (declared in the profile) must be present alongside any derived keys -- the native files provide the file-count reference and the derived path template.
