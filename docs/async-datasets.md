# Asynchronous Datasets

Asynchronous datasets model the reality of multi-sensor recording rigs: each sensor fires at its own rate, producing a separate stream of timestamped files. apairo merges these streams into a single timestamp-ordered timeline.

---

## Core concept

`ds[i]` returns **one event** -- one scan, one image, one IMU reading -- at its position in the merged timeline. Only the sensor that produced event `i` is populated in `sample.data`.

```python
sample = ds[0]
print(sample.timestamp)            # float -- seconds since epoch
print(list(sample.data.keys()))    # ["velodyne_0"]  -- exactly one key
print(sample.data["velodyne_0"].shape)
```

This is different from synchronous datasets where all requested modalities are always present. In async, iterate with `if "velodyne_0" in sample.data` to branch on sensor type.

If you want complete multi-channel frames instead of raw events, resample the dataset onto a reference clock with [`synchronize()`](#synchronizing-async-sync) -- the result behaves exactly like a synchronous dataset.

---

## TartanKittiDataset

`TartanKittiDataset` handles TartanDrive v2 recordings. It auto-detects whether the path is a single sequence or a root directory containing several.

### Single sequence

```python
from apairo import TartanKittiDataset

ds = TartanKittiDataset("/data/tartan/2024-01-01_forest", keys=["velodyne_0", "cmd"])
print(len(ds))        # total events across all loaded channels
print(ds.keys)        # ["cmd", "velodyne_0"]
```

### Root directory (multiple sequences)

```python
ds = TartanKittiDataset("/data/tartan", keys=["velodyne_0"])
print(len(ds))        # sum across all sequences
print(len(ds.sequences))    # number of sequences
```

### Lazy initialisation

Omit `keys` to inspect before loading anything:

```python
ds = TartanKittiDataset("/data/tartan/2024-01-01_forest")   # no loaders built yet
print(ds.available)          # frozenset of channels in .apairo
ds.keys = ["velodyne_0"]     # build loaders on demand
ds.keys = "all"              # or load every available channel
```

### describe()

`describe()` gives a human-readable breakdown of what is available without loading any data:

```python
TartanKittiDataset.describe("/data/tartan/2024-01-01_forest")
```

```
TartanKittiDataset -- 2024-01-01_forest
──────────────────────────────────────────────────
Raw channels
  present  : cmd, image_left, velodyne_0
  missing  : depth_left, imu

Preprocessed channels
  trav_label           npys   <- timestamps from velodyne_0  sources: ['velodyne_0']
```

---

## Auto-discovery and .apairo

On the first load of a new sequence, `TartanKittiDataset` scans the directory for known channel subdirectories and writes a `.apairo` sidecar:

```yaml
version: 1
channels:
  cmd:
    has_timestamps: true
    loader: npy
  velodyne_0:
    has_timestamps: true
    loader: npys
```

Subsequent loads read from `.apairo` and skip the scan. You can inspect or edit this file directly -- it is the authoritative record of what is available.

---

## register_channel

To manually register a channel (without running a preprocessor), use `register_channel`:

```python
from apairo import TartanKittiDataset

TartanKittiDataset.register_channel(
    "/data/tartan/2024-01-01_forest",
    key="my_channel",
    loader="npys",
    timestamps_from="velodyne_0",   # share velodyne timestamps
    sources=["velodyne_0"],         # provenance metadata
)
```

After registration, the channel is available as a loadable key:

```python
ds = TartanKittiDataset("/data/tartan/2024-01-01_forest", keys=["velodyne_0", "my_channel"])
```

`register_channel` is called automatically at the end of every `run_preprocess` call -- you only need it for manually placed files.

---

## Synchronizing: async → sync

The event timeline is the honest representation of a recording, but training
usually wants complete frames. `synchronize()` resamples the dataset onto a
single reference clock and returns a **synchronous view**: index `i` is the
*i*-th frame of the reference channel, with every other channel matched by
timestamp.

```python
ds = TartanKittiDataset(seq_dir, keys=["velodyne_0", "image_left", "cmd"])

ds_sync = ds.synchronize(
    reference="velodyne_0",   # default: the lowest-frequency channel
    method="latest",          # "latest" (zero-order hold) or "nearest"
    tolerance=0.05,           # drop frames with no match within ±50 ms
)

sample = ds_sync[0]
sample.data.keys()    # all three channels, always
sample.timestamp      # timestamp of the reference frame
```

- **`method="latest"`** -- each channel contributes its last event with
  `t <= t_ref` (what a live system would have seen at that instant).
- **`method="nearest"`** -- each channel contributes the event closest in
  time, past or future (best alignment for offline training).
- **`tolerance`** -- reference frames where any channel has no event within
  the window are dropped, so every sample is guaranteed fresh.

The matching is a pure index computation (one `searchsorted` per channel) --
nothing is read from disk until access. Inspect the alignment with
`ds_sync.frame_indices` and `ds_sync.time_offsets(key)`.

### External clocks: fixed-rate and distance-based resampling

The reference does not have to be a channel -- pass an **array of
timestamps** to resample onto any clock. Because the view is cheap to build,
changing the rate is a one-line edit, not a re-extraction:

```python
import numpy as np
from apairo.utils import clock_from_distance

t0, t1 = ds.timestamps["velodyne_0"][[0, -1]]

# Fixed rate: one frame every 100 ms
ds_10hz = ds.synchronize(reference=np.arange(t0, t1, 0.1))

# Spatial: one frame every 0.5 m travelled, from the odometry stream
odom = ds.loaders["odom"]
clock = clock_from_distance(ds.timestamps["odom"], odom_xy, step=0.5)
ds_spatial = ds.synchronize(reference=clock)
```

`clock_from_distance` ticks along the cumulative path length, so static
periods (robot not moving) produce no frames at all -- no separate trimming
step needed.

### Interpolating continuous channels

Matching picks an *existing* event; for continuous signals — poses, IMU,
commands — you often want the value *at* the reference instant instead.
Pass per-channel strategies as a dict; channels whose strategy is an
`Interpolator` are synthesized from their two bracketing events
(implementations live in
[apairo_transform](https://github.com/apairo-robotics/apairo_transform)):

```python
from apairo_transform.interp import LinearInterp, Se3Interp

ds_sync = ds.synchronize(
    reference="velodyne_0",
    method={
        "gicp_poses": Se3Interp(),     # slerp rotation + lerp translation
        "cmd":        LinearInterp(),  # linear blend
    },                                  # unlisted channels -> "latest"
    tolerance=0.5,
)

ds_sync[0].data["gicp_poses"]   # pose at exactly ds_sync[0].timestamp
ds_sync.time_offsets("gicp_poses")   # zeros: synthesized at the tick
```

Rules: ticks not bracketed by two events are dropped; exact matches return
the stored value untouched; with `tolerance`, *both* bracketing events must
lie within tolerance. Custom interpolators subclass `apairo.Interpolator` —
a single method `(t, t0, v0, t1, v1) -> value`.

### In-memory streams — `StreamDataset`

Data does not have to live on disk. `StreamDataset` builds an asynchronous
dataset from timestamped items already in memory — decoded ROS messages, a
live queue, arrays — and gives them the full apairo API, `synchronize()`
included. Items pass through untouched (they can be any Python objects):

```python
from apairo import StreamDataset

streams = StreamDataset({
    "image": (img_timestamps, img_msgs),
    "lidar": (lidar_timestamps, lidar_msgs),
    "odom":  (odom_timestamps, odom_msgs),
})

frames = streams.synchronize(reference=clock, method="latest")
frames[0].data   # {"image": msg, "lidar": msg, "odom": msg}
```

This is the bridge used to put apairo inside an existing extraction
pipeline: decode the bag as before, wrap the messages, and the temporal
matching logic disappears into `synchronize()`.

### Custom matching strategies

`method` also accepts a callable `(channel_ts, ref_ts) -> indices` returning,
for each reference tick, the event index to use (negative = no match, the
frame is dropped):

```python
def latest_within_100ms(ts: np.ndarray, ref_ts: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(ts, ref_ts, side="right") - 1
    idx[np.abs(ts[np.clip(idx, 0, None)] - ref_ts) > 0.1] = -1
    return idx

ds_sync = ds.synchronize(method=latest_within_100ms)
```

This is the extension point for exotic alignment strategies -- no subclassing
required.

On a root-level dataset, each sequence is synchronized on its own clock and
the results are concatenated -- timestamps are never compared across
recordings.

### The full API follows

The view is synchronous, so everything that works on synchronous datasets
works here -- including per-channel filtering and map-style `DataLoader` with
shuffling:

```python
from torch.utils.data import DataLoader

ds_train = (
    ds.synchronize(reference="velodyne_0", tolerance=0.05)
    .filter("velodyne_0", lambda pts: pts.shape[0] > 1000)
    .transform("velodyne_0", RangeFilter(max=50.0))
)

loader = DataLoader(ds_train, batch_size=8, shuffle=True, num_workers=4,
                    collate_fn=my_collate)
```

!!! note "Transforms and synchronize()"
    `synchronize()` reads channel data directly from the loaders -- transforms
    registered on the *async* parent are not applied. Register transforms on
    the synchronized view instead, as above.

---

## KittiDataset

`KittiDataset` is the base class for any KITTI-layout dataset -- one subdirectory per channel, each with a `timestamps.txt` and data files in a format declared by a loader profile YAML.

```python
from apairo import KittiDataset

ds = KittiDataset(
    directory="/data/my_recording",
    keys=["lidar", "imu"],
    dataset_profile="/path/to/my_profile.yaml",
)
```

**Profile YAML format:**

```yaml
# my_profile.yaml -- maps channel name -> loader type
lidar: npys
imu: npy
camera: img
```

**Extending KittiDataset:**

To create a reusable class for your dataset, subclass `KittiDataset` and `ConfigurableDataset`, and hardcode the profile path:

```python
from pathlib import Path
from apairo.dataset.kitti import KittiDataset
from apairo.core.configurable_dataset import ConfigurableDataset

_PROFILE = Path(__file__).parent / "my_profile.yaml"


class MyDataset(KittiDataset, ConfigurableDataset):
    def __init__(self, directory, keys=None):
        super().__init__(directory=directory, keys=keys or [], dataset_profile=_PROFILE)

    def _bootstrap_config(self, sequence_dir):
        # Return initial .apairo content when file doesn't exist yet
        return {
            "version": 1,
            "channels": {
                "lidar": {"loader": "npys", "has_timestamps": True},
            },
        }
```

---

## Expected directory layout

```
<sequence_dir>/
  .apairo               <- created automatically on first load
  velodyne_0/
    000000.bin
    000001.bin
    ...
    timestamps.txt
  image_left/
    000000.png
    ...
    timestamps.txt
  cmd/
    cmd.npy             <- single stacked file (NPYLoader)
    timestamps.txt
```
