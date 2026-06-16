# Frames & Transforms

Robotics data lives in **coordinate frames** (a lidar scan in the `lidar` frame,
a pose as `odom → base_link`, …). apairo's job is to *describe* that frame
information so downstream tools can use it — **not** to compute geometry.

## Where the line is

A capability becomes apairo's only where there is **one canonical way** to
represent the data; anything with **many ways** is either a pluggable interface
or lives downstream. Transforms split cleanly along this axis:

| Sub-capability | How many ways? | Home |
|---|---|---|
| **Describe** a transform (which frame a channel is in; static vs dynamic edges) | many (file formats, conventions) | **apairo** holds the *canonical representation*; format readers are pluggable |
| **Interpolate** a dynamic edge at time *t* | several (slerp, linear, …) | `Interpolator` interface (core) + impls in `apairo_transform` |
| **Apply** a transform (compose 4×4, move points) | one (canonical) | bounded; the spatial sibling of [`synchronize`](async-datasets.md) |
| **Calibrate** / register / SLAM | many (algorithms) | **out of scope** — `apairo_transform` or downstream |

apairo today implements the **describe** axis. Geometric *application* and,
above all, *calibration* are deliberately left out: a dataset loader exposes
frames and poses cleanly so a calibration or fusion tool — elsewhere — can
consume them.

## Canonical representation

### Per-channel frame

Each channel may declare the frame its data is expressed in, in
`.apairo/channels.yaml`:

```yaml
channels:
  lidar:
    loader: npys
    kind: raw
    has_timestamps: true
    frame: lidar          # the data in this channel lives in the `lidar` frame
```

Set it with `register_raw_channel(..., frame="lidar")` /
`register_channel(..., frame=...)`. `apairo status` shows a `frame` column when
any channel declares one (and carries it in `--json`).

### Static transforms — extrinsics

Fixed sensor mounting (e.g. `base_link → lidar`) is **calibration**. Datasets
expose it through the `calibration` property, keyed `"<from>_to_<to>"` with 4×4
homogeneous matrices:

```python
ds.calibration   # {"base_link_to_lidar": np.ndarray(4, 4), ...}
```

### Dynamic transforms — pose channels

A time-varying transform (`odom → base_link`, `map → odom`) is just a **pose
channel**: a timestamped series of poses, loaded like any other channel. It is
the same data your odometry node already publishes (e.g. `xyz + quaternion`).
Such a channel declares the edge it provides in `.apairo/channels.yaml`:

```yaml
channels:
  odom__base_link:
    loader: npy
    kind: raw
    has_timestamps: true
    transform: {parent: odom, child: base_link, format: t_xyz_q_xyzw}
```

`apairo-extractor` populates this automatically when it extracts a `/tf` topic
(one channel per parent→child edge; `/tf_static` edges carry `static: true`).
Set it manually with `register_raw_channel(..., transform={"parent": ..., "child": ...})`.
`apairo status` shows the edge (`← tf odom→base_link`).

**Looking it up** at an arbitrary time — composing the tree, interpolating with
`Se3Interp` — is the natural next step, but it is a geometric *verb*: it belongs
with `apairo_transform`, not in the core loader.

!!! note "What apairo does not do"
    `apairo` does not compose transform trees, apply transforms to point clouds,
    or estimate extrinsics. Those are geometric/algorithmic operations for
    `apairo_transform` or a downstream tool — they consume the frame information
    apairo exposes here.
