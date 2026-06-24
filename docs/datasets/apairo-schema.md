# The `.apairo` schema (version 1)

apairo records what it knows about a dataset in a small `.apairo/` directory next
to the data. This page is the **contract**: the on-disk format is `version: 1`
and stable. Do not confuse it with a [dataset *profile*](yaml-profiles.md)
(`rellis.yaml` …), which describes a dataset *class* shipped inside apairo; the
`.apairo/` sidecars describe one dataset *on disk*.

```
<root>/.apairo/
  channels.yaml      # the channel registry (per sequence directory)
  dataset.yaml       # root manifest -- optional
  calibration.yaml   # static extrinsics -- optional
```

Only `channels.yaml` is required for a directory to be a loadable apairo
sequence. `dataset.yaml` and `calibration.yaml` are **optional** — a dataset with
no extrinsics simply has no `calibration.yaml`.

## Compatibility policy

Validation is **tolerant**. An unknown field is reported as a warning and
otherwise ignored, so a sidecar written by a newer apairo still loads on an older
one. Every file carries a `version` (currently `1`); a different version is
flagged. Validate with `verify_config`, `verify_manifest`, `verify_calibration`
(or, from the shell, `apairo status` surfaces channel issues).

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
```

| Field | Required | Meaning |
|---|---|---|
| `kind` | yes | `raw` (on-disk modality) or `preprocess` (derived/persisted). |
| `loader` | yes | Storage format: `npy` (one stacked file, row per frame), `npys` (one file per frame), `bin`, `img`, `zarr`. |
| `timestamps_from` | no | The channel whose timestamps this one shares (provenance). |
| `sources` | no | Channels this one was derived from (provenance). |
| `frame` | no | Coordinate frame the data is expressed in. Descriptive only — apairo never applies transforms. |
| `transform` | no | Declares the channel *is* a transform stream: `{parent, child, [static], [format]}`. Descriptive only. |
| `alias` | no | Public name the channel loads under (the directory keeps its real name). Must be unique and must not shadow a real channel directory. |

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

## `calibration.yaml` (static extrinsics, optional)

Time-independent transforms. apairo **exposes** them (via `dataset.calibration` →
`{"<parent>_to_<child>": 4x4 float64}`) and **resolves** any pair of connected
frames with `dataset.calibration.get_tf(source, target)`; it never *applies* the
result to data — that is `apairo_transform`'s job.

```yaml
version: 1
transforms:
  lidar_to_camera:
    parent: lidar
    child: camera
    matrix: [[...4x4...]]
```

Each entry needs `parent`, `child`, and a 4×4 `matrix`. Write them with
`register_static_transform(root, parent, child, matrix)`.
