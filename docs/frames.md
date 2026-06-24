# Frames & Calibration

Robotics data lives in **coordinate frames** (a lidar scan in the `lidar` frame,
a pose as `odom -> base_link`, ...). apairo's job is to *describe* that frame
information -- and to *resolve* the one part of it that is unambiguous, the static
transform tree -- so downstream tools can use it. It does not *apply* geometry to
your data; that is data-dependent and lives in `apairo_transform`.

## Where the line is

A capability becomes apairo's only where there is **one canonical way** to
represent the data; anything with **many ways** is either a pluggable interface
or lives downstream. Transforms split cleanly along this axis:

| Sub-capability | How many ways? | Home |
|---|---|---|
| **Describe** which frame a channel is in; static vs dynamic edges | many (file formats, conventions) | **apairo** holds the *canonical representation*; format readers are pluggable |
| **Resolve** the static tree — the transform between two fixed frames | one (a graph walk) | **apairo** — `ds.calibration.get_tf(source, target)` |
| **Interpolate** a dynamic edge at time *t* | several (slerp, linear, …) | `Interpolator` interface (core) + impls in `apairo_transform` |
| **Apply** to data — move points, re-express poses/normals | many (data-dependent) | `apairo_transform` (`TransformPoints`, …) |
| **Calibrate** / register / SLAM | many (algorithms) | **out of scope** — downstream |

apairo **describes** the frame graph and **resolves** its static part: there is
exactly one way to compose fixed edges, so that walk lives in the core. What it
leaves out is everything with *many* right answers — applying a transform to data
(points vs poses vs normals), interpolating a dynamic edge at time *t*, and
estimating extrinsics in the first place. Those consume what apairo exposes; they
live in `apairo_transform` or downstream.

## Canonical representation

### Per-channel frame

Each channel may declare the frame its data is expressed in, in
`.apairo/channels.yaml`:

```yaml
channels:
  lidar:
    loader: npys
    kind: raw
    frame: lidar          # the data in this channel lives in the `lidar` frame
```

Set it with `register_raw_channel(..., frame="lidar")` /
`register_channel(..., frame=...)`. `apairo status` shows a `frame` column when
any channel declares one (and carries it in `--json`).

### Static transforms — extrinsics

Fixed sensor mounting (e.g. `base_link → lidar`) is **calibration** — it does
not vary with time, so it does *not* belong in a per-frame channel. Datasets
expose it through the `calibration` property, keyed `"<parent>_to_<child>"` with
4×4 homogeneous matrices:

```python
ds.calibration   # {"base_link_to_lidar": np.ndarray(4, 4), ...}
```

On disk it is a single `.apairo/calibration.yaml` (one entry per edge), written
with `register_static_transform(root, parent, child, matrix)`. `apairo-extractor`
populates it from `/tf_static`: every static edge becomes one calibration entry
rather than its own channel — so a tree with dozens of fixed mounts stays one
small file, not dozens of directories.

**Resolving** any two connected frames is the one geometric step that *is*
canonical — a walk over that tree — so it lives on the calibration object itself:

```python
T = ds.calibration.get_tf("lidar", "base_link")   # T_base_link_from_lidar; p_base = T @ p_lidar
```

`get_tf(source, target)` returns the 4×4 mapping a point from `source` into
`target`, composing and inverting edges as needed (identity when the frames
match, `KeyError` when no static path connects them). *Applying* that matrix to
real data — a cloud, a pose, surface normals — is the data-dependent step, and so
it is `apairo_transform`'s job, not the core's:

```python
from apairo_transform import TransformPoints
ds.transform("lidar", TransformPoints(T))
```

### Dynamic transforms — pose channels

A time-varying transform (`odom → base_link`, `map → odom`) is just a **pose
channel**: a timestamped series of poses, loaded like any other channel. It is
the same data your odometry node already publishes (e.g. `xyz + quaternion`).
Such a channel declares the edge it provides in `.apairo/channels.yaml`:

```yaml
channels:
  tf__odom__base_link:
    loader: npy
    kind: raw
    transform: {parent: odom, child: base_link, source: /tf, format: t_xyz_q_xyzw}
```

`apairo-extractor` populates this automatically when it extracts a `/tf` topic:
it demultiplexes the message into one pose channel per edge, named
`<source>__<parent>__<child>`. Naming by source is deliberate — it is a faithful
dump, so the *same* edge coming from different sources (e.g. two odometry
stacks) is preserved as **distinct channels** you can compare, never merged or
dropped. (Static transforms go to `calibration`, above — only time-varying
edges become channels.) Set it manually with
`register_raw_channel(..., transform={"parent": ..., "child": ...})`;
`apairo status` shows the edge (`<- tf odom→base_link`).

**Looking it up** at an arbitrary time — interpolating the edge with `Se3Interp`,
then chaining it onto the static tree — is the natural next step, but the
interpolation has several valid forms: it belongs with `apairo_transform`, not
the core loader.

!!! note "What apairo does not do"
    `apairo` resolves the *static* tree, but it does not apply transforms to data,
    interpolate a dynamic edge at an arbitrary time, or estimate extrinsics. Those
    have many valid implementations — they belong to `apairo_transform` or a
    downstream tool, which consume the frame information apairo exposes here.
