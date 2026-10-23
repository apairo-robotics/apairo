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

If you want complete multi-channel frames instead of raw events, resample the dataset onto a reference clock with [`synchronize()`](synchronization.md) -- the result behaves exactly like a synchronous dataset.

### Frame provenance — `frame_info`

`sample.data` tells you *which* channel event `i` is, but not *which frame of that channel* it is, nor (on a root) which sequence. `frame_info(i)` answers all three -- the public accessor for tooling that maps a flat index back to its origin (a visualizer logging "lidar frame 1813", a splitter, an exporter), instead of reaching into private timeline state:

```python
ref = ds.frame_info(i)        # FrameRef(sequence, channel, row)
ref.sequence                  # the sequence the frame belongs to
ref.channel                   # the channel that fired (one event = one channel)
ref.row                       # frame index within that channel
```

- On a **synchronous** dataset a frame is *all* channels at the same row, so `channel` is `None` and `row` is the frame index.
- The companion array properties cover the whole dataset at once: **`frame_sequence_ids`** (the sequence each frame belongs to), **`frame_stems`** (the data-file stem backing each frame, or the zero-padded row for stacked channels), and **`frame_channel_ids`** (the channel that produced each frame). These also make a frozen, file-based split work on a generic `RawDataset` -- filter to a saved `(sequence, stem)` set, the same mechanism profiled datasets use.

```python
# Freeze an eval split to a set of (sequence, stem) pairs and reload it:
keep = {(s, t) for s, t in zip(ds.frame_sequence_ids, ds.frame_stems) if (s, t) in eval_set}
eval_ds = ds.filter([i for i, (s, t) in
                     enumerate(zip(ds.frame_sequence_ids, ds.frame_stems)) if (s, t) in keep])

# Restrict a raw timeline to one channel's events -- no more per-frame scans:
camera_only = ds.filter(np.nonzero(ds.frame_channel_ids == "camera")[0])
```

`frame_channel_ids` raises `AttributeError` wherever a frame has no single origin channel: synchronous datasets (a frame is every channel) and `SynchronizedView` (a synchronised frame composites one event per channel -- use `frame_indices` there for real per-channel provenance).

`frame_info` is read-only and delegates through `filter`/`select` views (the index is remapped for you).

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

### Choosing channels

Omitting `keys` opens every channel the sequence declares. Narrow the set when
you open it, or change it later -- loaders are built for the keys you set:

```python
ds = TartanKittiDataset("/data/tartan/2024-01-01_forest", keys=["velodyne_0"])
print(ds.available)          # frozenset of channels in .apairo
ds.keys = ["velodyne_0", "cmd"]   # rebuild on another set
ds.keys = "all"              # or every available channel
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

## RawDataset

`RawDataset` is the **profile-free** member of the family. Where
`TartanKittiDataset` pins a fixed channel set (the TartanDrive profile),
`RawDataset` takes its channels — and each channel's storage format — entirely
from `.apairo/channels.yaml`. There is no code to write per dataset: whatever
the sidecar declares is what loads. This makes it the loader for
[apairo_extractor](https://github.com/apairo-robotics/apairo_extractor) output,
which writes that sidecar at extraction time.

```python
from apairo import RawDataset

# A single sequence (has .apairo/channels.yaml)
ds = RawDataset("/data/my_dataset/seq_a", keys=["lidar", "imu"])

# A whole dataset root (has .apairo/dataset.yaml, or sequence subdirectories)
ds = RawDataset("/data/my_dataset")
print(ds.name)            # from the manifest, else the directory name
print(ds.sequence_ids)    # ["seq_a", "seq_b", ...]
print(len(ds.sequences))  # per-sequence RawDataset instances
print(len(ds))            # flat event count across all sequences
```

The single-sequence vs. root distinction is auto-detected, exactly like
`TartanKittiDataset`. `keys=None` loads every channel declared in the sidecar.

### Format-agnostic

The on-disk layout (one directory per channel, each with its own
`timestamps.txt`) is fixed; the *format* of each channel is not. The loader named
in `channels.yaml` decides — `npy`, `npys`, `bin`, `img`, or `zarr` — and they
can be mixed freely within one dataset:

```yaml
# seq_a/.apairo/channels.yaml
version: 1
channels:
  lidar: { loader: npys, kind: raw }   # one .npy per frame
  imu:   { loader: npy,  kind: raw }   # one stacked .npy
  gps:   { loader: zarr, kind: raw }   # a zarr store
```

A `zarr` channel directory *is* the zarr array store, with `timestamps.txt`
placed beside the chunks.

### Dataset root manifest

A root may carry an optional `.apairo/dataset.yaml` recording the dataset name
and the sequence order. When absent, sequences are discovered and sorted by name.

```yaml
# <root>/.apairo/dataset.yaml
version: 1
name: my_dataset
sequences: [seq_a, seq_b, seq_c]   # load order
```

### Initializing an existing directory

For a directory that does not yet have a sidecar, `RawDataset.init` scans it and
writes one (loaders inferred from file extensions / a detected zarr store):

```python
RawDataset.init("/data/my_dataset/seq_a")   # writes .apairo/channels.yaml
ds = RawDataset("/data/my_dataset/seq_a")
```

Everything the scan cannot know - a filename `key` regex, a `pcd` `fields`
contract, an `alias` - belongs in a
[declaration](datasets/apairo-schema.md#apairoyaml-the-declaration-human-owned):
`<seq>/apairo.yaml`, or a file outside the tree passed as `declare=` (a root
propagates it to every sequence):

```python
ds = RawDataset("/data/my_dataset", declare="eval/my_dataset.yaml")
```

`RawDataset` is asynchronous, so the [`synchronize()`](synchronization.md)
section below applies unchanged; on a root, each sequence is synchronized on its
own clock and the results concatenated.

---

## Auto-discovery and .apairo

On the first load of a new sequence, `TartanKittiDataset` scans the directory for known channel subdirectories and writes a `.apairo` sidecar:

```yaml
version: 1
channels:
  cmd:
    loader: npy
  velodyne_0:
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

## Aliasing channels

`RawDataset` takes its channel names straight from the directories on disk, which
for `apairo-extractor` output are ROS topic names (`ouster_points`,
`dlio_odom_node_odom`). An **alias** gives a channel a clean public name without
renaming anything on disk: the directory keeps its real name, but the dataset
loads and exposes it under the alias. This brings the profile-free `RawDataset`
the canonical-naming ergonomics a profiled dataset gets from its layout -- and
the naming lives in the dataset's metadata, not scattered across your scripts.
(`set_alias` records it in the machine registry; an
[`apairo.yaml` declaration](datasets/apairo-schema.md#apairoyaml-the-declaration-human-owned)
declares the same `alias:` field under your ownership - preferred for aliases
you author and version.)

```python
import apairo

apairo.set_alias("/data/barakuda/seq_a", "ouster_points", "lidar")
apairo.set_alias("/data/barakuda/seq_a", "dlio_odom_node_odom", "pose")

ds = apairo.RawDataset("/data/barakuda/seq_a", keys=["lidar", "pose"])
ds[0].data["lidar"]          # the ouster_points directory, exposed as "lidar"
ds.timestamps["lidar"]       # keyed by the public name too
```

- The real directory name is still accepted in `keys=[...]`, but the channel is
  always exposed under its alias (the alias is the canonical public name).
- `apairo.set_alias(seq_dir, channel, None)` clears the alias.
- From the shell: [`apairo alias <channel> <alias>`](cli.md#apairo-alias)
  (root-aware), and `apairo status` surfaces the aliases.
- It is recorded per channel in `.apairo/channels.yaml` as
  `{loader: ..., alias: lidar}`.

---

## StreamDataset — in-memory streams

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

frames = streams.synchronize(reference=clock, method="previous")
frames[0].data   # {"image": msg, "lidar": msg, "odom": msg}
```

This is the bridge used to put apairo inside an existing extraction
pipeline: decode the bag as before, wrap the messages, and the temporal
matching logic disappears into `synchronize()`.

---

## AsyncLayoutDataset (the base class)

`AsyncLayoutDataset` is the abstract base of the asynchronous family — the
on-disk *layout* primitive: one subdirectory per channel, each with a
`timestamps.txt` and data files in a format known to the loader registry
(`npys`, `npy`, `bin`, `img`, `zarr`). It describes *how* channels are stored,
never *which* channels exist; the channel set is per-instance state, read from
`.apairo/channels.yaml` or an explicit profile.

It is an internal base class — you rarely use it directly. Pick the member that
matches your channel set:

- **Dynamic channels** (whatever a recording happens to contain, e.g.
  `apairo-extractor` output) → [`RawDataset`](#rawdataset), which reads the
  channel set from `.apairo` with no profile.
- **Fixed channels** (a known dataset family) → subclass it with a profile, as
  `TartanKittiDataset` does.

```python
from apairo.dataset.async_layout import AsyncLayoutDataset

ds = AsyncLayoutDataset(
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
gps: zarr
```

**Extending it for a fixed-channel dataset:**

Subclass `AsyncLayoutDataset` and `ConfigurableDataset` and hardcode the profile
path (for multi-sequence root support, also mix in
`apairo.core.root_sequence.RootSequenceMixin`, as `TartanKittiDataset` does):

```python
from pathlib import Path
from apairo.dataset.async_layout import AsyncLayoutDataset
from apairo.core.configurable_dataset import ConfigurableDataset

_PROFILE = Path(__file__).parent / "my_profile.yaml"


class MyDataset(AsyncLayoutDataset, ConfigurableDataset):
    def __init__(self, directory, keys=None):
        super().__init__(directory=directory, keys=keys or [], dataset_profile=_PROFILE)

    def _bootstrap_config(self, sequence_dir):
        # Return initial .apairo content when file doesn't exist yet
        return {
            "version": 1,
            "channels": {
                "lidar": {"loader": "npys", "kind": "raw"},
            },
        }
```

If your channels are not fixed, prefer `RawDataset` — no subclass needed.

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
