# Bring your own dataset

apairo reads a dataset as a set of **channels** — one directory per channel,
holding per-frame files (or a single stacked file) in a format the loader
registry understands (`npy`, `npys`, `bin`, `img`, `zarr`). Turning *your*
directories into a loadable dataset comes down to three small questions per
channel. Two have always been implicit; this page makes all three explicit — and
shows how the **`key`** and **`order`** fields let a channel carry its own
alignment clock and enumeration policy right in its filenames, with **nothing
ever written** into your data tree.

This is the entry point for the asynchronous family: the profile-free
[`RawDataset`](../async-datasets.md#rawdataset) (channels read straight from
`.apairo/channels.yaml`) and its fixed-channel subclasses such as
`TartanKittiDataset`.

---

## The three contracts

Every channel answers three questions. apairo ships a sensible default for each,
so a plain directory of zero-padded files needs no configuration at all.

| Contract | Question it answers | Built-in default |
|---|---|---|
| **order** | how are this channel's frames listed and ordered? | files matching the frame-file convention (`is_frame_file`), sorted numerically |
| **load** | how is one frame decoded to a numpy array? | the loader named in `channels.yaml` (`npy` / `npys` / `bin` / `img` / `zarr`) |
| **key** | what is each frame's alignment key — the value `synchronize()` matches on? | the channel's own `timestamps.txt`, else a borrowed one (`timestamps_from`), else the frame's position |

`order` and `load` were always there; `key` used to be hardwired to
`timestamps.txt`. The additive change is that **`key`** and **`order`** are now
declarable per channel — and both default to today's behavior, so everything you
already load keeps loading identically.

!!! note "Position is the default key"
    A channel with no `key` and no `timestamps.txt` is aligned by frame
    **position**: frame *i* of every such channel is the same sample (the
    synchronous, KITTI-style case). A `key` you provide lets channels of
    different rates — or sparse subsets — align by timestamp or index instead.
    *(Collapsing the synchronous `ProfiledDataset` family into this same
    position-as-default key is a separate, larger step — see
    [Not in scope](#not-in-scope).)*

---

## Nothing is written — keys are read-time, in memory

The key is computed **in memory, at read time. apairo never writes into your
dataset tree.** A filename-keyed channel derives its clock from the filenames it
already has, so it needs no `timestamps.txt` sidecar at all. This keeps loading
strictly read-only — it works on a read-only mount, a shared cluster volume, an
`ro` container. Materializing a self-contained clock belongs to the *write* side
(`export`, the extractor), never to reading.

---

## The `key` field

Add `key:` to a channel in `channels.yaml`. Two forms.

### `key: {name: <regex>}` — parse the key from the filename

The regex is matched against each file's **stem** (its name without the
extension); the capture groups are combined into a number:

- **one group** → an integer index or a float — `frame(\d+)-` on
  `frame000123-...` yields `123`;
- **two groups** → `'<int>.<frac>'` by the default rule
  `float('.'.join(groups))` — a Rellis `frame000123-1581624652_750` stem under
  `frame\d+-(\d+)_(\d+)` yields `1581624652.750`.

```yaml
camera: {loader: img, key: {name: 'frame\d+-(\d+)_(\d+)'}}   # timestamp from the filename
seg:    {loader: img, key: {name: 'frame(\d+)-'}}            # integer index from the filename
```

#### `scale:` — an explicit unit combine

When joining the groups as a decimal string is not what you want — the
fractional part is not zero-padded, or the groups are in different units — give
an explicit `scale`, one factor per group, summed as
`sum(int(group_i) * scale_i)`:

```yaml
camera:
  loader: img
  key: {name: 'frame\d+-(\d+)_(\d+)', scale: [1, 0.001]}   # seconds + milliseconds
```

`<sec>` counts as seconds and `<ms>` as milliseconds, so `1581624652_750`
becomes `1581624652 + 750 * 0.001 = 1581624652.750` regardless of how the
millisecond field is padded.

#### `units:` — the readable form of `scale`

Raw `scale` factors are terse. When every capture group is a **time field**, name
the units instead — `units:` compiles to `scale` (`s` → 1, `ms` → 1e-3,
`us` → 1e-6, `ns` → 1e-9):

```yaml
camera: {loader: img, key: {name: 'frame\d+-(\d+)_(\d+)', units: [s, ms]}}
```

reads exactly as `scale: [1, 0.001]` but says what it means. `units` and `scale`
are mutually exclusive.

!!! note "Heterogeneous names — a timestamp *and* an index"
    `units` only combines **time** fields. A stem like `camera_<sec>_<frame-index>`
    mixes a clock with a counter, and an index is not a duration — don't fold it
    into the key. Capture **only the key field** and let `order` handle the rest:

    ```yaml
    camera:
      loader: img
      key:   {name: 'camera_(\d+)_\d+', units: [s]}   # the seconds are the key
      order: {name: 'camera_\d+_(\d+)'}                # the index only sorts
    ```

    The index never pollutes the alignment key. Frames that share a second then
    share a key — `synchronize()` treats them as simultaneous; if you need
    sub-second ordering *from* the index, a `_key_providers` callable is the
    escape hatch.

### `key: {file: <name>}` — read the key from a sidecar

Read the key array from a named file inside the channel directory, one float per
line, in frame order. This generalizes `timestamps.txt` to any filename:

```yaml
imu: {loader: npy, key: {file: stamps.txt}}
```

---

## The `order` field

`order` is a **separate contract** from `key`: it decides which files are
enumerated and in what order, independently of how the key is computed.

`order: {name: <regex>}` — the files whose stem matches, sorted
**lexicographically** (which is frame order for zero-padded names).

You rarely set it, because it defaults sensibly:

- absent **and** `key: {name: ...}` is set → `order` reuses the key's `name`
  regex;
- absent otherwise → the default frame-file convention (`is_frame_file`, which
  reserves `_` for suffixes and sorts numerically).

!!! warning "When `order` is required"
    The default convention **reserves `_`** for channel suffixes and rejects any
    stem that carries one. A channel whose filenames embed an underscore the
    default would skip — a Rellis `<epoch>_<ms>` stem — must therefore be
    enumerated by a regex. Setting `key: {name: ...}` covers this automatically
    (its regex doubles as the order). Set `order: {name: ...}` explicitly only
    when the key comes from elsewhere (a `file:` key, or a callable) *and* the
    filenames still carry a `_`.

---

## How it feeds `synchronize()`

The alignment engine is unchanged. Once a channel's `key` populates
`ds.timestamps[channel]`,
[`synchronize()`](../async-datasets.md#synchronizing-async-sync) aligns it
exactly as always — `reference=`, `method=` (`nearest` / `previous` / `next`),
`tolerance=`. Sparse subsets fall out for free: a channel with no key at a
reference tick simply contributes nothing there.

```python
ds = apairo.RawDataset(root, keys=["camera", "seg"])
view = ds.synchronize(reference="camera", method="nearest", tolerance=0.0)
# every frame carries "camera"; "seg" attaches only where a label exists
```

---

## Escape hatch — a callable per channel (the 10%)

Anything the DSL can't express, hand in as a callable from a subclass. Before
`super().__init__()`, populate either provider dict — they are checked **before**
the YAML specs:

```python
class MyDataset(RawDataset):
    def __init__(self, directory, keys=None):
        self._key_providers = {
            "camera": lambda files: np.array([parse_epoch(f) for f in files]),
        }
        self._order_providers = {
            "camera": lambda directory: sorted(exotic_listing(directory)),
        }
        super().__init__(directory, keys=keys)
```

- `self._key_providers[channel]` — `callable(files: list[str]) -> np.ndarray`,
  the channel's key array.
- `self._order_providers[channel]` — `callable(directory: str) -> list[str]`,
  the ordered filenames.

Either can be set on its own; the YAML `key` / `order` specs cover the remaining
channels.

---

## Worked example — Rellis-3D camera

Rellis-3D historically loaded only lidar + labels: its camera could not be added.
The files are named `frame000123-<epoch>_<ms>.jpg` — an index identical to the
lidar, dressed with a timestamp and carrying a `_` the default convention rejects
— and its hand-labels are a ~half-rate subset, neither expressible by the
equal-count positional reader. Two lines in `channels.yaml` express both, with no
subclass and no transcode:

```yaml
pylon_camera_node:          {loader: img, key: {name: 'frame\d+-(\d+)_(\d+)'}}   # <epoch>_<ms> timestamp
pylon_camera_node_label_id: {loader: img, key: {name: 'frame\d+-(\d+)_(\d+)'}}   # SAME timestamp -> label lands on its camera tick
```

- **`pylon_camera_node`** — the key is the `<epoch>_<ms>` timestamp parsed from
  each stem. Because `key: {name}` is set, that same regex enumerates the
  underscore-bearing files, so `order` needs nothing.
- **`pylon_camera_node_label_id`** — keyed by the **same** timestamp. The labels
  exist on a subset of frames, each sharing its camera frame's `<epoch>_<ms>`, so
  `synchronize(..., tolerance=0.0)` lands every label on its exact camera tick and
  keeps only the labeled frames. Both channels must key in the **same space** — an
  index-keyed label against an epoch-keyed camera would never match.

```python
ds = apairo.RawDataset(rellis_seq,
                       keys=["pylon_camera_node", "pylon_camera_node_label_id"])
view = ds.synchronize(reference="pylon_camera_node", method="nearest", tolerance=0.0)
```

On real Rellis-3D this parses **2847 camera frames at 10 Hz** straight from the
filenames (median Δt 100.0 ms, monotonic) and synchronizes the **1200 sparse
image-labels** onto the dense camera — with **zero writes** into the Rellis tree.

---

## Not in scope

This is the additive `key` / `order` contract for the **asynchronous** family.
Unifying the **synchronous** `ProfiledDataset` family (SemanticKITTI, Rellis,
GOOSE) behind the same key — making position-as-default one setting of a single
order/load/key contract shared by both families, instead of two separate dataset
families — is a larger architectural step and is **not** part of this feature.
See `IDEAS.md`.