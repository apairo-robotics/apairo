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
usage: apairo [-h] {init,status} ...

Inspect and initialize apairo datasets.

  init     write .apairo sidecars by scanning a directory
  status   show what a dataset directory contains
```

`PATH` defaults to the current directory, so you can `cd` into a dataset and run
the commands with no arguments.

---

## `apairo init`

Scan a directory and write its `.apairo` sidecar(s). **Root-aware**: it
auto-detects whether the path is a single sequence or a dataset root.

```
apairo init [PATH] [--name NAME] [--force] [--as CLASS]
```

- **Sequence** (its sub-directories hold data files) → writes
  `.apairo/channels.yaml`, inferring each channel's loader from the files on
  disk (`npy`, `npys`, `bin`, `img`, `zarr`).
- **Root** (its sub-directories are sequences) → initializes each sequence, then
  writes the root `.apairo/dataset.yaml` (name + sequence order + channel union).

It is **idempotent** and **non-destructive**: by default it *merges* (adds newly
detected channels, leaves existing declarations untouched). Pass `--force` to
rebuild from scratch.

```bash
# Initialize / repair a whole dataset produced by apairo-extractor
apairo init /data/my_dataset --name my_dataset
```

```
✓ wrote .apairo/dataset.yaml
RawDataset — my_dataset   (root · 2 sequences)
────────────────────────────────────────────────────
sequences   seq_a, seq_b
raw         imu (npy), lidar (npys)
preprocess  —
events      14
issues      none
```

!!! tip "Repairing existing datasets"
    If you have data that was laid out before its `.apairo` files existed (e.g.
    an older extraction), `apairo init <dir>` reconstructs them in place — no
    re-extraction or re-download needed. The result loads directly with
    [`RawDataset`](async-datasets.md#rawdataset).

| Option | Meaning |
|---|---|
| `--name NAME` | Dataset name for the root manifest (default: directory name) |
| `--force` | Rebuild from scratch instead of merging |
| `--as CLASS` | Interpret with a specific dataset class (default: `RawDataset`) |

---

## `apairo status`

Report what a dataset directory contains, without loading any heavy data (frame
counts come from each channel's `timestamps.txt`).

```
apairo status [PATH] [--json] [--as CLASS]
```

It distinguishes **tracked** channels (declared in `.apairo`) from **untracked**
ones (channel directories present on disk but not yet registered), and surfaces
any consistency issues found by `verify_config`.

```bash
apairo status /data/my_dataset
```

```
RawDataset — my_dataset   (root · 2 sequences)
────────────────────────────────────────────────────
sequences   seq_a, seq_b
raw         imu (npy), lidar (npys)
preprocess  trav_gt (npys)
untracked   seq_a/segmentation   ← run `apairo add`
events      14
issues      none
```

`--json` emits the same information as a machine-readable object, suitable for
scripts and CI:

```bash
apairo status /data/my_dataset --json
```

```json
{
  "name": "my_dataset",
  "kind": "root",
  "sequences": ["seq_a", "seq_b"],
  "raw": {"imu": "npy", "lidar": "npys"},
  "preprocess": {"trav_gt": "npys"},
  "untracked": ["seq_a/segmentation"],
  "events": 14,
  "issues": []
}
```

A directory that is neither a sequence nor a dataset root exits non-zero with a
hint to run `apairo init`.

---

## The `.apairo` layout

Both commands read and write the same on-disk convention:

```
<root>/                         # dataset root
  .apairo/dataset.yaml          # name + sequence order (written by `init` on a root)
  seq_a/
    .apairo/channels.yaml       # channel -> loader/kind/timestamps (per sequence)
    lidar/  000000.npy ... timestamps.txt
    imu/    imu.npy             timestamps.txt
  seq_b/ ...
```

This is exactly what [`RawDataset`](async-datasets.md#rawdataset) loads, so
`init` → `status` → load is one coherent workflow.

!!! note "Planned commands"
    `apairo add` (register an untracked channel) and `apairo check` (consistency
    check, non-zero exit on problems) are planned follow-ups. `status` already
    surfaces the untracked channels that `add` will act on.
