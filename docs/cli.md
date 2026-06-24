# Command Line

`apairo` ships a small command-line tool to **inspect** and **initialize**
datasets from the terminal. It is a thin wrapper over the library (no
third-party dependencies) and operates on the `.apairo` sidecars that describe a
dataset on disk.

Installing apairo puts the command on your `PATH`:

```bash
pip install apairo
apairo --help
```

```
usage: apairo [-h] {init,status,alias} ...

Inspect and initialize apairo datasets.

  init     write .apairo sidecars by scanning a directory
  status   show what a dataset directory contains
  alias    expose a channel under a clean public name
```

`PATH` defaults to the current directory, so you can `cd` into a dataset and run
the commands with no arguments.

### Ecosystem commands

Beyond the built-ins (`init`, `status`), `apairo` is also a **dispatcher** for
the wider ecosystem. Installed tools register a subcommand via the
`apairo.cli_plugins` entry-point group, so they appear as `apairo <tool>`:

```bash
apairo extractor -i bags/ -o dataset/ -t /lidar /tf   # = apairo-extractor
```

This is plugin discovery, not a dependency: `apairo` never imports its tools, it
just dispatches to whatever is installed (so there is no circular dependency
between the packages). `apairo --help` lists the discovered commands. A package
exposes one like this:

```toml
[project.entry-points."apairo.cli_plugins"]
extractor = "apairo_extractor.cli:main"
```

---

## `apairo init`

Scan a directory and write its `.apairo` sidecar(s). **Root-aware**: it
auto-detects whether the path is a single sequence or a dataset root.

```
apairo init [PATH] [--name NAME] [--force] [--as CLASS]
```

- **Sequence** (its sub-directories hold data files) -> writes
  `.apairo/channels.yaml`, inferring each channel's loader from the files on
  disk (`npy`, `npys`, `bin`, `img`, `zarr`).
- **Root** (its sub-directories are sequences) -> initializes each sequence, then
  writes the root `.apairo/dataset.yaml` (name + sequence order + channel union).

It is **idempotent** and **non-destructive**: by default it *merges* (adds newly
detected channels, leaves existing declarations untouched). Pass `--force` to
rebuild from scratch.

```bash
# Initialize / repair a whole dataset produced by apairo-extractor
apairo init /data/my_dataset --name my_dataset
```

```
wrote .apairo/dataset.yaml
RawDataset - my_dataset   (root, 2 sequences)
----------------------------------------------------
sequences   seq_a, seq_b
raw         imu (npy), lidar (npys)
preprocess  -
events      14
issues      none
```

!!! tip "Repairing existing datasets"
    If you have data that was laid out before its `.apairo` files existed (e.g.
    an older extraction), `apairo init <dir>` reconstructs them in place - no
    re-extraction or re-download needed. The result loads directly with
    [`RawDataset`](async-datasets.md#rawdataset).

| Option | Meaning |
|---|---|
| `--name NAME` | Dataset name for the root manifest (default: directory name) |
| `--force` | Rebuild from scratch instead of merging |
| `--as CLASS` | Interpret with a specific dataset class (default: `RawDataset`) |

`--as <Class>` maps a registered dataset's profile (e.g. `Rellis3DDataset`,
`Goose3DDataset`, `SemanticKittiDataset`) onto the directories on disk, writing
canonical channel names into `.apairo/channels.yaml`. The dataset **class** is
recorded in `.apairo/dataset.yaml`, so a later `apairo status` dispatches through
the same profile instead of falling back to the generic reading.

```bash
apairo init /data/RELLIS --as Rellis3DDataset
```

---

## `apairo status`

Report what a dataset directory contains, without loading any heavy data (frame
counts come from each channel's `timestamps.txt`).

```
apairo status [PATH] [-s ID] [--json] [--show-tf]
```

It distinguishes **tracked** channels (declared in `.apairo`) from **untracked**
ones (channel directories present on disk but not yet registered), and surfaces
any consistency issues found by `verify_config`.

If the directory was initialized as a specific dataset class (`init --as
<Class>`, recorded in `.apairo/dataset.yaml`), `status` dispatches **through that
profile**: it names the class, lists the sequences, and resolves canonical
channel names (`lidar`) to their real nested directories -- rather than the
profile-unaware generic reading.

| Option | Meaning |
|---|---|
| `-s, --sequence ID` | Per-channel detail for one sequence, addressed by id from the root |
| `--json` | Machine-readable output (same information as the table) |
| `--show-tf` | Include the transform layer: static calibration + dynamic `tf` channels (hidden by default) |

### On a dataset root - summary

```bash
apairo status /data/my_dataset
```

```
RawDataset - my_dataset   (root, 2 sequences)
----------------------------------------------------
sequences   seq_a, seq_b
raw         imu (npy), lidar (npys)
preprocess  trav_gt (npys)
untracked   seq_a/segmentation   <- run `apairo init` to register
events      14
issues      none
```

For a directory initialized as a dataset class, the header names that class and
the channels show their **canonical** names (the profile resolves the nested
directories on disk):

```bash
apairo init /data/RELLIS --as Rellis3DDataset
apairo status /data/RELLIS
```

```
Rellis3DDataset - RELLIS   (root, 2 sequences)
----------------------------------------------------
sequences   00000, 00001
raw         labels (bin), lidar (bin)
preprocess  trav_gt (npys)
events      18
issues      none
```

### On a single sequence - per-channel detail

Point `status` at a sequence to get a per-channel table -- or, from the root, pass
`-s <id>` to address a sequence by **id** without naming its directory:

```bash
apairo status /data/my_dataset/seq_a   # by path
apairo status /data/my_dataset -s seq_a # by id, from the root (equivalent)
```

`-s` is the only way to inspect a **profiled** dataset's sequence with its
canonical channel names: the raw data lives in nested, profile-mapped directories
(e.g. `Rellis-3D/00000/os1_cloud_node_kitti_bin/`) that a plain
`status <nested-dir>` cannot interpret. `-s` resolves them through the profile.

Every column is cheap:
`frames`, `rate` and `span` come from each channel's `timestamps.txt`, and
`shape`/`dtype` are read from the `.npy` header via `mmap` (no data is loaded).

`span` is shown **relative to the earliest timestamp** of the sequence (printed
once on the `start` line), so absolute epoch timestamps stay readable and
per-channel start offsets are easy to compare.

```bash
apairo status /data/my_dataset/seq_a
```

```
RawDataset - seq_a   (sequence)
----------------------------------------------------------------------
start       1779893201.02s   (span shown relative to this)
channel       kind        loader  frames  rate     span         shape
imu           raw         npy     200     20.0 Hz  0.00-9.95s   (6) float64
lidar         raw         npys    100     10.0 Hz  0.03-9.93s   (4, 3) float32
trav_gt       preprocess  npys    100     10.0 Hz  0.03-9.93s   (1,) uint8   <- from lidar
segmentation  untracked   npys     98      -          -         (2, 2) uint8 <- run `apairo init`
events      400
issues      none
```

(Rate and span are not comparable across recordings, so the root view stays a
summary and the per-channel detail lives at the sequence level.)

`--json` emits the same information as a machine-readable object, suitable for
scripts and CI. On a root it carries the summary; on a sequence it carries the
full per-channel detail:

```bash
apairo status /data/my_dataset/seq_a --json
```

JSON carries the **absolute** spans (ground truth) plus the `start` reference, so
consumers can reconstruct the relative view shown in the table:

```json
{
  "name": "seq_a",
  "kind": "sequence",
  "start": 1779893201.02,
  "channels": {
    "lidar": {
      "kind": "raw", "loader": "npys", "frames": 100,
      "rate_hz": 10.0, "span": [1779893201.05, 1779893210.95],
      "shape": [4, 3], "dtype": "float32"
    },
    "imu": {
      "kind": "raw", "loader": "npy", "frames": 200,
      "rate_hz": 20.0, "span": [1779893201.02, 1779893210.97],
      "shape": [6], "dtype": "float64"
    }
  },
  "untracked": {},
  "events": 400,
  "issues": []
}
```

A directory that is neither a sequence nor a dataset root exits non-zero with a
hint to run `apairo init`.

---

## `apairo alias`

Give a channel a clean **public name**. The on-disk directory keeps its real
name, but the dataset loads and exposes it under the alias -- so a
[`RawDataset`](async-datasets.md#rawdataset) built from `apairo-extractor` output
(channels named after ROS topics like `ouster_points`, `dlio_odom_node_odom`)
reads with the same canonical names a profiled dataset would use.

```
apairo alias CHANNEL [ALIAS] [--path PATH] [--remove] [--force]
```

```bash
apairo alias ouster_points lidar --path /data/barakuda   # ouster_points -> lidar
apairo alias dlio_odom_node_odom pose --path /data/barakuda
```

It is **root-aware**: pointed at a dataset root, it sets the alias in every
sequence that declares the channel; pointed at a single sequence, only that one.

A public name must be unique: an alias that already belongs to another channel,
or that shadows a real directory name, is **rejected** (it would make the dataset
unloadable). Pass `--force` to *reassign* an alias from its current holder (which
is then left unaliased and reported); a clash with a real directory name is never
reassignable.

```python
# now the same names everywhere -- script stays readable, naming lives in .apairo
ds = apairo.RawDataset("/data/barakuda", keys=["lidar", "pose"])
ds[0].data["lidar"]        # the ouster_points directory, exposed as "lidar"
```

`status` shows aliases: an `aliases` line on the root summary, and the per-channel
table prints the alias first with the directory in parentheses
(`lidar (ouster_points)`). Clear an alias with `--remove` (omit `ALIAS`).

The alias also works from Python:
[`apairo.set_alias(seq_dir, channel, alias)`](async-datasets.md#aliasing-channels),
or `RawDataset.register_raw_channel(seq_dir, channel, loader, alias=...)`.

| Option | Meaning |
|---|---|
| `--path PATH` | Dataset directory (default: `.`); root-aware |
| `--remove` | Clear the channel's alias instead of setting it |
| `--force` | Reassign the alias even if another channel holds it (that channel is left unaliased) |

---

## `apairo check`

Validate a dataset's `.apairo` sidecars against the [version-1
schema](datasets/apairo-schema.md) and report any issues. Exits non-zero when
there is at least one issue, so it drops straight into CI.

```bash
apairo check [PATH] [--json]
```

```text
$ apairo check /data/my_dataset
OK -- no issues
```

It is profile-aware (the same reading as `status`) and covers all three files:
`channels.yaml`, the optional `dataset.yaml` manifest, and the optional
`calibration.yaml`. Validation is tolerant -- an unknown field is reported as a
warning, not a hard error.

---

## The `.apairo` layout

Both commands read and write the same on-disk convention:

```
<root>/                         # dataset root
  .apairo/dataset.yaml          # identity: dataset class, name, sequence order
  seq_a/
    .apairo/channels.yaml       # channel -> loader/kind/timestamps (per sequence)
    lidar/  000000.npy ... timestamps.txt
    imu/    imu.npy             timestamps.txt
  seq_b/ ...
```

`dataset.yaml` is the root **manifest** (identity: which dataset `class` produced
the layout, plus name / sequence order); `channels.yaml` is the per-directory
**channel registry**. A profiled dataset (`init --as <Class>`) writes both at the
root -- the manifest records the class so `status` can dispatch through the
profile.

This is exactly what [`RawDataset`](async-datasets.md#rawdataset) loads, so
`init` -> `status` -> load is one coherent workflow.

!!! note "`apairo add`"
    `apairo add` (register an untracked channel) is deferred to post-1.0.
    `status` surfaces the untracked channels, and re-running `apairo init` (or
    `register_raw_channel` / `register_channel` from Python) registers them.
