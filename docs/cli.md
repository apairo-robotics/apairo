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

### On a dataset root — summary

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

### On a single sequence — per-channel detail

Point `status` at a sequence to get a per-channel table. Every column is cheap:
`frames`, `rate` and `span` come from each channel's `timestamps.txt`, and
`shape`/`dtype` are read from the `.npy` header via `mmap` (no data is loaded).

```bash
apairo status /data/my_dataset/seq_a
```

```
RawDataset — seq_a   (sequence)
──────────────────────────────────────────────────────────────────────
channel       kind        loader  frames  rate     span         shape
imu           raw         npy     200     20.0 Hz  0.00–9.95s   (6) float64
lidar         raw         npys    100     10.0 Hz  0.00–9.90s   (4, 3) float32
trav_gt       preprocess  npys    100     10.0 Hz  0.00–9.90s   (1,) uint8   ← from lidar
segmentation  untracked   npys     98      —          —         (2, 2) uint8 ← run `apairo add`
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

```json
{
  "name": "seq_a",
  "kind": "sequence",
  "channels": {
    "lidar": {
      "kind": "raw", "loader": "npys", "frames": 100,
      "rate_hz": 10.0, "span": [0.0, 9.9], "shape": [4, 3], "dtype": "float32"
    },
    "imu": {
      "kind": "raw", "loader": "npy", "frames": 200,
      "rate_hz": 20.0, "span": [0.0, 9.95], "shape": [6], "dtype": "float64"
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
