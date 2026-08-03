# The `.apairo` schema (version 1)

apairo records what it knows about a dataset in a small `.apairo/` directory next
to the data. This page is the **contract**: the on-disk format is `version: 1`
and stable. Do not confuse it with a [dataset *profile*](yaml-profiles.md)
(`rellis.yaml` …), which describes a dataset *class* shipped inside apairo; the
`.apairo/` sidecars describe one dataset *on disk*.

```
<root>/apairo.yaml   # the human declaration -- optional, never written by apairo
<root>/.apairo/
  channels.yaml      # the channel registry (per sequence directory)
  dataset.yaml       # root manifest -- optional
  calibration.yaml   # static extrinsics -- optional
```

Only `channels.yaml` is required for a directory to be a loadable apairo
sequence. `dataset.yaml` and `calibration.yaml` are **optional** — a dataset with
no extrinsics simply has no `calibration.yaml`. The split between `apairo.yaml`
and `.apairo/` is by **owner**: the declaration is yours (apairo reads it and
never writes it), the sidecar directory is the machine's — see
[`apairo.yaml`](#apairoyaml--the-declaration-human-owned) below.

## Compatibility policy

Validation is **tolerant**. An unknown field is reported as a warning and
otherwise ignored, so a sidecar written by a newer apairo still loads on an older
one. Every file carries a `version` (currently `1`); a different version is
flagged. Validate with `verify_config`, `verify_manifest`, `verify_calibration`,
`verify_declaration` (or, from the shell, `apairo status` surfaces channel and
declaration issues).

## `channels.yaml`

```yaml
version: 1
channels:
  ouster_points:           # key == the on-disk directory name
    kind: raw              # "raw" | "preprocess"           (required)
    loader: bin            # npy | npys | bin | img | zarr   (required)
    alias: lidar           # public name exposed at load time          (optional)
    frame: ego             # coordinate frame, descriptive only        (optional)
  trav_gt:
    kind: preprocess
    loader: npys
    timestamps_from: lidar # channel whose timestamps this one shares  (optional)
    sources: [labels]      # provenance: channels it was derived from   (optional)
  gicp_odom:
    kind: raw
    loader: npy
    transform:             # this channel *is* a coordinate transform   (optional)
      parent: odom         #   (required inside transform)
      child: base_link     #   (required inside transform)
      static: false        #   (optional)
      format: t_xyz_q_xyzw #   (optional)
  gicp_poses:              # two stacked arrays colocated in one directory
    kind: raw
    loader: npy
    array_file: poses.npy  # the exact .npy this channel loads           (optional)
  gicp_valid_mask:
    kind: raw
    loader: npy
    directory: gicp_poses  # share gicp_poses/ rather than own a dir     (optional)
    array_file: valid_mask.npy
  ouster_points:           # vendor PCL clouds, read in place
    kind: raw
    loader: pcd
    frame: os_sensor
    fields: [x, y, z, intensity]  # the channel's field contract         (optional)
```

| Field | Required | Meaning |
|---|---|---|
| `kind` | yes | `raw` (on-disk modality) or `preprocess` (derived/persisted). |
| `loader` | yes | Storage format: `npy` (one stacked file, row per frame), `npys` (one file per frame), `bin`, `img`, `zarr`, `pcd`. |
| `timestamps_from` | no | The channel whose timestamps this one shares (provenance). |
| `sources` | no | Channels this one was derived from (provenance). |
| `frame` | no | Coordinate frame the data is expressed in. Descriptive only — apairo never applies transforms. |
| `transform` | no | Declares the channel *is* a transform stream: `{parent, child, [static], [format]}`. Descriptive only. |
| `alias` | no | Public name the channel loads under (the directory keeps its real name). Must be unique and must not shadow a real channel directory. |
| `directory` | no | On-disk subdirectory the channel's files live in, when different from its key — lets a channel share another channel's directory. Defaults to the key. |
| `suffix` | no | Per-frame colocation: load only `<frame_stem>_<suffix>.npy` from `directory` (e.g. `velodyne_0/000000_intensity.npy` beside `000000.npy`). `npys` only. |
| `array_file` | no | Whole-array colocation: the exact stacked `.npy` this channel loads from `directory`, when it holds more than one (e.g. `valid_mask.npy` beside `poses.npy`). `npy` only. |
| `fields` | no | The field contract of a `pcd` channel, e.g. `[x, y, z, intensity]`. A PCD header is self-describing, so two frames may declare different fields; naming them here selects those columns in that order, making the channel's width a declared property rather than a per-file accident. A frame missing one raises, naming both sets. Omitted, every field the file declares is returned in header order. `pcd` only. |

## `apairo.yaml` — the declaration (human-owned)

`channels.yaml` holds two kinds of knowledge with different owners: what the
machine can regenerate (discovered `raw` channels, `preprocess` provenance
written by `run_preprocess`) and what only a human can know — a filename
[`key`](bring-your-own-dataset.md#the-key-field) regex, a `pcd`
`fields` contract, an `alias`. The declaration gives the human half a home of
its own, **outside** the dot-directory:

* apairo **never writes** `apairo.yaml`. `init` (including `--overwrite`) and
  `run_preprocess` only touch `.apairo/`, so a registry rebuild cannot destroy
  a declaration. Version it in git next to your eval code.
* It uses the same version-1 schema as `channels.yaml`, **minus machine
  provenance**: `kind: preprocess`, `sources` and `recipe` are refused — with
  an error, not a warning.
* It overlays the registry **per channel and per field**. Declaring a `key`
  for a channel does not repeat the `loader` the registry already knows; a
  channel that exists only in the declaration must name its `loader`.
* The bootstrap scan respects it: stems explained by a declared `key`/`order`
  regex are not fanned out into suffixed sub-channels.

```yaml
# <seq>/apairo.yaml -- versioned by you, never touched by apairo
version: 1
channels:
  "pcd (2)":                       # key == the on-disk directory name
    loader: pcd
    alias: lidar                   # exposed as "lidar" at load time
    fields: [x, y, z, intensity]   # stable (N, 4) float32
    key: {name: '(\d{16,})$', units: [ns]}   # clock parsed from the stems
```

A declaration can also live entirely outside the data tree and be passed at
load time — the read-only-mount case:

```python
ds = apairo.RawDataset(root, declare="eval/barakuda.yaml")
```

Per-field precedence, highest first: `declare=` > `<seq>/apairo.yaml` >
`.apairo/channels.yaml`. On a dataset root, `declare=` applies to every
sequence.

## `dataset.yaml` (root manifest, optional)

Identity for a dataset **root** (the parent of several sequence directories).

```yaml
version: 1
class: Rellis3DDataset   # the profiled class that produced the layout (profiled roots)
name: my_dataset         # optional display name
sequences: [00000, 00001]  # generic roots: sequence order
channels: {lidar: {kind: raw}}  # generic roots: channel roll-up
```

`class` is written by a profiled `init --as <Class>` so `apairo status` can
dispatch through that profile; `name` / `sequences` / `channels` describe a
generic (`RawDataset`) root.

## `calibration.yaml` (static rig configuration, optional)

Time-independent rig configuration: extrinsic transforms and camera
intrinsics. apairo **exposes** them (via `dataset.calibration`) and
**resolves** any pair of connected frames with
`dataset.calibration.get_tf(source, target)`; it never *applies* the result to
data — that is `apairo_transform`'s job.

```yaml
version: 1
transforms:
  lidar_to_camera:
    parent: lidar
    child: camera
    matrix: [[...4x4...]]
cameras:
  multisense_left:          # the camera's frame (CameraInfo frame_id)
    K: [[...3x3...]]
    D: [k1, k2, p1, p2, k3] # optional -- omit for a rectified image
    distortion_model: plumb_bob
    width: 1024
    height: 544
    R: [[...3x3...]]        # optional (stereo-rectified rigs)
    P: [[...3x4...]]        # optional (stereo-rectified rigs)
```

Each transform needs `parent`, `child`, and a 4×4 `matrix`; write them with
`register_static_transform(root, parent, child, matrix)`. Each camera needs a
3×3 `K` (field names mirror ROS `CameraInfo`); write them with
`register_intrinsics(root, camera, K=..., ...)` and read them back with
`dataset.calibration.get_intrinsics(camera)`.
