# Write a Format Plugin

apairo reads every channel through a **format**. Seven are built in: `npy`,
`npys`, `bin`, `img`, `zarr`, `pcd` and `csv`. If your data is stored some other
way, you don't need to change apairo. You write a small class in a package of
your own, and apairo finds it through an entry point. Once that package is
installed, `loader: <name>` works everywhere a built-in name does:

- `apairo init` detects the channel;
- `apairo declare` scaffolds its clock;
- `apairo check` validates it;
- `apairo status` describes it;
- `RawDataset` loads it, and `synchronize()` aligns it with the other channels.

## What the core keeps, what a format answers

| The core, the same for every format | The format, about its own bytes |
| --- | --- |
| `.apairo/channels.yaml`, `apairo.yaml`, `--declare` | which directories hold it (`detect`) |
| `timestamps.txt`, `timestamps_from` | which files are its frames (`extensions`, `matches`) |
| the `key: {name: ...}` and `key: {file: ...}` clocks, the `order` regex | the loader that decodes a frame (`loader`, `open`) |
| suffixed variants, colocated channels (`directory: "."`) | clock forms its data provides (`key_forms`, `clock`) |
| the clock checks, the timeline, `synchronize()`, views, caching | the channel fields it reads (`fields`, `validate`) |
| | what `status` shows and `declare` suggests (`facts`, `declare_hints`) |

A per-frame format gets a filename clock, an `order` regex and the `status`
facts without writing any code for them.

## A complete plugin: `.xyz` point clouds

The `.xyz` format, which some scanners and photogrammetry tools export, stores
one ASCII point cloud per frame, with one `x y z` point per line. Here is the
whole plugin:

```python title="apairo_xyz.py"
--8<-- "examples/format_plugin/apairo_xyz.py"
```

```toml title="pyproject.toml"
--8<-- "examples/format_plugin/pyproject.toml"
```

Install it with `pip install -e .` and point apairo at a sequence whose
`lidar/` holds `.xyz` files named by their nanosecond stamp:

```console
$ apairo init seq/
$ apairo declare seq/
$ cat seq/apairo.yaml
...
  lidar:
    loader: xyz
    # alias: <public name at load time>
    key: {name: '(\d+)$', units: [ns]}   # 19-digit epoch guessed from the stems -- verify
$ apairo status seq/
```

The scaffolded `key` line comes from the core, because the format is per-frame.
The same example runs in apairo's test suite (`test/core/test_format_contract.py`):
it goes through `init`, `declare`, `check`, `status`, loading and
`synchronize()`.

## Test it against the contract

`apairo.testing.check_format` checks a format on a small sample channel. It
reports every breach at once:

```python
from apairo.testing import check_format
from apairo_xyz import XyzFormat


def test_xyz_follows_the_contract(tmp_path):
    for k in range(3):
        (tmp_path / f"{k:06d}.xyz").write_text("0 0 0\n1 1 1\n")
    check_format(XyzFormat(), tmp_path)
```

The check covers the following:

- **Attributes.** They are well-formed, and a `key_form` comes with a `clock()`.
- **Detection.** The sample directory is detected, and an empty one is not.
- **Loading.** The loader opens the sample and reads its first and last frames.
- **File lists.** A per-frame loader exposes `files` and honors the `files` it
  is given.
- **Status.** `facts()` agrees with the loader.
- **Scaffold and check.** `declare_hints()` and `validate()` answer in the
  expected shape.

The built-in formats pass the same check.

## The contract

`Format` is a class. You set its attributes and override only what differs
from the defaults. The defaults describe a per-frame format: one file per
frame, recognized by its extension, opened as `loader(directory)` or
`loader(directory, files=[...])`.

| Attribute | Default | Meaning |
| --- | --- | --- |
| `name` | (required) | The name channels declare: `loader: <name>`. Letters, digits, `_` and `-`. |
| `loader` | (required) | An `AbstractLoader` subclass. `len(loader)` is the number of frames and `loader[i]` is frame `i`. A per-frame loader also sets `files`, one filename per frame. |
| `extensions` | `{}` | The data-file suffixes, in lower case and with the dot. |
| `per_frame` | `True` | One file per frame. A stacked format (`False`) holds every frame in one object and takes no filename `key`/`order`. |
| `one_channel_per_file` | `False` | Each file is a channel of its own, like a table. |
| `suffixes` | `False` | Frames may have suffixed variants (`000000_intensity.npy`), which become sibling channels. |
| `fields` | `{}` | The channel fields the format reads, which may include fields of its own. `check` accepts them on its channels and flags them on others. |
| `key_forms` | `{}` | Clock forms the data provides, such as `{"column"}` for `csv`. Declared as `key: {<form>: ...}`. |
| `priority` | `50` | The detection order, lowest first. The built-ins use `zarr` 10, `bin` 20, `pcd` 30, `img` 40, `npys` 50, `npy` 51 and `csv` 90. |

| Method | Default | Override when |
| --- | --- | --- |
| `matches(path)` | the suffix is in `extensions` | a suffix alone is ambiguous, like `.txt` tables |
| `detect(directory)` | any data file | the format is a store directory (`zarr`) or depends on a file count |
| `open(directory, meta, files)` | `loader(dir)` or `loader(dir, files=files)` | the loader needs channel fields from `meta` |
| `clock(loader, spec, label)` | raises an error | you declare `key_forms` |
| `facts(directory, meta, files)` | open, then read frame 0 | reading frame 0 is not cheap |
| `declare_hints(directory, meta)` | none | a scaffold line would help, such as the fields a header lists |
| `validate(key, meta, storage_dir)` | no issues | the format has fields of its own |

`files` is the list of frame filenames the core resolved from an `order` or
`key` regex, or for a suffixed variant. It is `None` when the format should list
its own frames.

`validate` checks only the fields that are present and never requires one.
`check` reads `channels.yaml` and the declaration separately, and a declaration
may supply a field that the registry entry lacks. If a field is still missing at
load time, `open()` refuses the channel and names the field.

## A stacked format with its own clock

Some formats carry their clock inside the data, such as a column, a JSON
member or a dataset in a container. Declare that clock as a key form and
compute it in `clock()`:

```python
class JsonlFormat(Format):
    name = "jsonl"
    loader = JsonlLoader            # frame i: the `values` member of line i
    extensions = frozenset({".jsonl"})
    per_frame = False
    fields = frozenset({"values"})  # a field of its own
    key_forms = frozenset({"member"})

    def open(self, directory, meta, files=None):
        return JsonlLoader(directory, values=meta.get("values", "v"))

    def clock(self, loader, spec, label):
        return np.array([r[spec["member"]] for r in loader.records], dtype=float)
```

```yaml
channels:
  joints:
    loader: jsonl
    values: q
    key: {member: stamp}
```

`check` knows `values` as a channel field and `member` as a key form. On an
`npys` channel it reports `key.member` as needing the `jsonl` loader.

## Rules

- **Registering.** An installed plugin registers through its entry point. A
  script or notebook can call `apairo.core.formats.register_format(MyFormat)`
  instead.
- **Names.** A built-in name cannot be replaced. A name that is already taken
  needs `register_format(..., replace=True)`.
- **Broken plugins.** A plugin that fails to load is skipped, with a warning
  that names it. One broken package never makes other datasets unreadable.
- **Heavy dependencies.** apairo imports every installed plugin when it first
  looks a format up. Import heavy libraries inside `open()` or the loader, not
  at the top of the module.
- **Reading only.** A format plugin adds a way to read data. Preprocessing
  outputs are still written by apairo's writers (`npys`, `npy`, `bin`, `zarr`,
  `img`).
