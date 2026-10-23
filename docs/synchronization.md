# Synchronization

An [asynchronous dataset](async-datasets.md) serves one event per index: one
scan, one image, one IMU reading.
The event timeline is the honest representation of a recording, but training
usually wants complete frames. `synchronize()` resamples the dataset onto a
single reference clock and returns a **synchronous view**: index `i` is the
*i*-th frame of the reference channel, with every other channel matched by
timestamp.

```python
ds = TartanKittiDataset(seq_dir, keys=["velodyne_0", "image_left", "cmd"])

ds_sync = ds.synchronize(
    reference="velodyne_0",   # default: the lowest-frequency channel
    method="previous",        # "previous" (zero-order hold), "next" or "nearest"
    tolerance=0.05,           # drop frames with no match within ±50 ms
)

sample = ds_sync[0]
sample.data.keys()    # all three channels, always
sample.timestamp      # timestamp of the reference frame
```

- **`method="previous"`** -- each channel contributes its last event with
  `t <= t_ref` (what a live system would have seen at that instant;
  `"latest"` is a deprecated alias).
- **`method="next"`** -- each channel contributes its first event with
  `t >= t_ref` (e.g. attach the result of a computation that lands after
  the tick).
- **`method="nearest"`** -- each channel contributes the event closest in
  time, past or future, ties favouring the earlier event (best alignment
  for offline training).
- **`tolerance`** -- reference frames where any channel has no event within
  the window are dropped, so every sample is guaranteed fresh.

The matching is a pure index computation (one `searchsorted` per channel) --
nothing is read from disk until access. Inspect the alignment with
`ds_sync.frame_indices` and `ds_sync.time_offsets(key)`.

## External clocks: fixed-rate and distance-based resampling

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
odom_xy = np.stack([odom[i][:2] for i in range(len(odom))])  # x, y per row
clock = clock_from_distance(ds.timestamps["odom"], odom_xy, step=0.5)
ds_spatial = ds.synchronize(reference=clock)
```

`clock_from_distance` ticks along the cumulative path length, so static
periods (robot not moving) produce no frames at all -- no separate trimming
step needed.

## Interpolating continuous channels

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
    },                                  # unlisted channels -> "previous"
    tolerance=0.5,
)

ds_sync[0].data["gicp_poses"]   # pose at exactly ds_sync[0].timestamp
ds_sync.time_offsets("gicp_poses")   # zeros: synthesized at the tick
```

Rules: ticks not bracketed by two events are dropped; exact matches return
the stored value untouched; with `tolerance`, *both* bracketing events must
lie within tolerance. Custom interpolators subclass `apairo.Interpolator` —
a single method `(t, t0, v0, t1, v1) -> value`.

## Custom matching strategies

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

## On a dataset root

On a root-level dataset, each sequence is synchronized on its own clock and
the results are concatenated -- timestamps are never compared across
recordings. For the same reason an external clock array is refused on a
root: synchronize the sequences one by one (`ds.sequences[i].synchronize(...)`)
and [concatenate](combining.md) the results.

## The full API follows

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

## Persisting a synchronization

`synchronize()` recomputes its matching every session. To reuse exactly the
same frames later, or to hand them to someone else, persist the view under a
name:

```python
ds_sync = ds.synchronize(reference="velodyne_0", method="nearest", tolerance=0.05)
ds_sync.persist("lidar_sync")       # into <sequence>/.apairo, no data copied

# later, or on another machine with the same data
ds_sync = RawDataset(seq).load_view("lidar_sync")
```

`persist` writes what the view *is*:

- the reference ticks it kept;
- one index array per channel;
- the method and the tolerance;
- a fingerprint of every source channel: its frame count and a hash of its
  clock.

`load_view` rebuilds the same synchronous dataset from those indices without
matching again. It refuses a view whose sources have changed since, and names
the channel. For example, a frame was added, a timestamp moved, a `latency`
was declared, or the sequence was opened through another declaration.
Recompute it with `persist(name, overwrite=True)`. On a root, each sequence
keeps its own view.

The matching itself is fast, a few milliseconds even for thousands of frames.
What a persisted view adds is reproducibility: the frames of a training set,
fixed by name, that can be checked against the data.

An interpolated channel is persisted with its bracketing pairs. The
interpolator is code, not state, so it is passed again when the view is
loaded: `load_view(name, interpolators={"gicp_poses": Se3Interp()})`.
A custom matching callable is not needed again, since its indices are kept.
`apairo status` lists a sequence's views, and `apairo check` validates their
registry. The format is in [the `.apairo` schema](datasets/apairo-schema.md#viewsyaml-persisted-synchronizations).

