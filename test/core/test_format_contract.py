"""The format contract: a new storage format is a plugin, not a core change.

Every format -- the built-in ones and a plugin -- answers the same questions
through :class:`apairo.core.formats.Format`; the core asks the registry and
never names a format. The proof is end to end: the ``xyz`` example plugin
(``examples/format_plugin``), installed through its entry point, is detected by
``init``, scaffolded by ``declare``, validated by ``check``, described by
``status`` and loaded and synchronized like a built-in channel -- with no line
of apairo changed. A stacked plugin brings its own clock form and channel
field the same way.
"""

from __future__ import annotations

import ast
import json
import logging
from importlib.metadata import EntryPoint
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import apairo.core.formats as registry
from apairo.cli import main
from apairo.core.abstract_loader import AbstractLoader
from apairo.core.config import verify_config, write_config
from apairo.core.formats import Format, get_format, register_format
from apairo.dataset.raw import RawDataset
from apairo.testing import check_format
from test.loader.test_pcd_loader import XYZI, write_binary

REPO = Path(__file__).resolve().parents[2]
T0_NS = 1785248347168248576
BUILTINS = ("npy", "npys", "bin", "img", "zarr", "pcd", "csv")


def _cli(args, capsys):
    try:
        code = main(args)
    except SystemExit as exc:
        code = exc.code
    return code, capsys.readouterr().out


@pytest.fixture
def plugins(monkeypatch):
    """Install formats through the ``apairo.formats`` entry point group, as
    ``pip install`` would; the registry is restored afterwards."""
    registry.format_names()  # built-ins in first
    monkeypatch.setattr(registry, "_FORMATS", dict(registry._FORMATS))
    monkeypatch.setattr(registry, "_plugins_loaded", False)
    monkeypatch.syspath_prepend(str(REPO / "examples" / "format_plugin"))

    def install(*specs: tuple[str, str]) -> None:
        eps = [
            EntryPoint(name, value, registry.ENTRY_POINT_GROUP) for name, value in specs
        ]
        monkeypatch.setattr(
            "importlib.metadata.entry_points",
            lambda group=None: [ep for ep in eps if ep.group == group],
        )
        monkeypatch.setattr(registry, "_plugins_loaded", False)

    return install


# ───────────────────────────── the built-in formats ───────────────────────────


def _sample(tmp_path: Path, name: str) -> tuple[Path, dict]:
    d = tmp_path / name
    d.mkdir()
    meta: dict = {}
    if name == "npys":
        for i in range(3):
            np.save(d / f"{i:06d}.npy", np.full((4, 3), i, np.float32))
    elif name == "npy":
        np.save(d / "poses.npy", np.arange(12.0).reshape(4, 3))
    elif name == "bin":
        for i in range(3):
            np.full((5, 4), i, np.float32).tofile(d / f"{i:06d}.bin")
    elif name == "img":
        for i in range(3):
            Image.fromarray(np.full((6, 8, 3), i, np.uint8)).save(d / f"{i:06d}.png")
    elif name == "pcd":
        for i in range(3):
            write_binary(d / f"{i:06d}.pcd", XYZI, [[0.0, 1.0, 2.0, float(i)]] * 2)
        meta = {"fields": ["x", "y", "z"]}
    elif name == "csv":
        (d / "imu.csv").write_text("t,ax,ay\n1.0,0,1\n1.1,2,3\n1.2,4,5\n")
        meta = {"array_file": "imu.csv", "key": {"column": "t"}}
    elif name == "zarr":
        zarr = pytest.importorskip("zarr")
        arr = zarr.open_array(str(d), mode="w", shape=(4, 3), dtype="float32")
        arr[:] = np.arange(12, dtype=np.float32).reshape(4, 3)
    return d, meta


@pytest.mark.parametrize("name", BUILTINS)
def test_every_builtin_format_follows_the_contract(tmp_path, name):
    directory, meta = _sample(tmp_path, name)
    check_format(get_format(name), directory, meta)


def test_the_kit_names_every_breach(tmp_path):
    class Loose(Format):
        name = "loose"
        loader = dict  # type: ignore[assignment]
        extensions = frozenset({"TXT"})
        key_forms = frozenset({"name"})

    with pytest.raises(AssertionError) as err:
        check_format(Loose(), tmp_path)
    message = str(err.value)
    assert "`loader` must be an AbstractLoader subclass" in message

    class Forgetful(Format):  # a key form, no clock()
        name = "forgetful"
        loader = AbstractLoader
        extensions = frozenset({"TXT"})
        key_forms = frozenset({"row", "name"})

    with pytest.raises(AssertionError) as err:
        check_format(Forgetful(), tmp_path)
    message = str(err.value)
    assert "lower-case and start with '.'" in message
    assert "may not claim the core's key fields ['name']" in message
    assert "needs `clock()` implemented" in message


def test_the_kit_catches_a_loader_that_ignores_files(tmp_path):
    from apairo.loader.npys_loader import NPYSLoader

    class Deaf(NPYSLoader):
        def __init__(self, directory, files=None):
            super().__init__(directory)  # drops what the core resolved

    class DeafFormat(Format):
        name = "deaf"
        loader = Deaf
        extensions = frozenset({".npy"})

    directory, _ = _sample(tmp_path, "npys")
    with pytest.raises(AssertionError, match="open\\(\\) ignores `files`"):
        check_format(DeafFormat, directory)


# ─────────────────────── a per-frame plugin, end to end ───────────────────────


def _xyz_sequence(seq: Path) -> Path:
    """``lidar``: four .xyz clouds named by a nanosecond stamp, no clock file;
    ``cam``: npys frames at 10 Hz with a timestamps.txt."""
    lidar = seq / "lidar"
    lidar.mkdir(parents=True)
    for k in range(4):
        rows = "\n".join(f"{k}.0 {j}.0 0.5" for j in range(5))
        (lidar / f"{T0_NS + k * 100_000_000}.xyz").write_text(rows + "\n")
    cam = seq / "cam"
    cam.mkdir()
    for i in range(4):
        np.save(cam / f"{i:06d}.npy", np.full(2, i, np.float32))
    np.savetxt(cam / "timestamps.txt", T0_NS * 1e-9 + np.arange(4) * 0.1, fmt="%.9f")
    return seq


def test_an_installed_plugin_is_a_channel_like_any_other(tmp_path, plugins, capsys):
    plugins(("xyz", "apairo_xyz:XyzFormat"))
    seq = _xyz_sequence(tmp_path / "seq")

    code, out = _cli(["init", str(seq)], capsys)
    assert code in (0, None), out
    code, out = _cli(["check", str(seq), "--json"], capsys)
    assert code == 1
    assert any("'lidar': no clock" in i for i in json.loads(out)["issues"])

    _cli(["declare", str(seq)], capsys)
    declaration = (seq / "apairo.yaml").read_text()
    assert "    loader: xyz" in declaration
    assert "    key: {name: '(\\d+)$', units: [ns]}" in declaration

    code, out = _cli(["check", str(seq)], capsys)
    assert code == 0, out
    _, out = _cli(["status", str(seq), "--json"], capsys)
    lidar = json.loads(out)["channels"]["lidar"]
    assert lidar["loader"] == "xyz"
    assert lidar["frames"] == 4
    assert lidar["shape"] == [5, 3]
    assert lidar["dtype"] == "float32"
    assert lidar["rate_hz"] == pytest.approx(10.0)

    ds = RawDataset(seq)
    view = ds.synchronize(reference="cam", method="nearest", tolerance=0.01)
    assert len(view) == 4
    np.testing.assert_allclose(view[2].data["lidar"][:, 0], 2.0)

    from apairo_xyz import XyzFormat

    check_format(XyzFormat(), seq / "lidar")


def test_a_preprocess_on_plugin_data_writes_the_plugin_format(tmp_path, plugins):
    """No output_loader: a crop of .xyz clouds is written as .xyz clouds, and a
    conversion into the plugin's format is a preprocess like any other."""
    from apairo.core.config import read_config
    from apairo.core.preprocessor import FramePreprocessor
    from apairo.preprocess import Convert

    plugins(("xyz", "apairo_xyz:XyzFormat"))
    seq = _xyz_sequence(tmp_path / "seq")
    (seq / "apairo.yaml").write_text(
        "version: 1\nchannels:\n  lidar:\n    loader: xyz\n"
        "    key: {name: '(\\d+)$', units: [ns]}\n"
    )
    RawDataset.init(seq)

    class _Crop(FramePreprocessor):
        output_key = "near"
        input_keys = ["lidar"]

        def __call__(self, sample):
            cloud = sample.data["lidar"]
            return cloud[cloud[:, 1] < 3]

    RawDataset.run_preprocess(_Crop(), seq)
    assert read_config(seq)["channels"]["near"]["loader"] == "xyz"
    assert len(list((seq / "near").glob("*.xyz"))) == 4
    ds = RawDataset(seq, keys=["near"])
    assert ds.loaders["near"][0].shape == (3, 3)

    clouds = seq / "cloud"
    clouds.mkdir()
    for i in range(2):
        np.save(clouds / f"{i:06d}.npy", np.full((2, 3), i, np.float32))
    np.savetxt(clouds / "timestamps.txt", [1.0, 2.0])
    RawDataset.init(seq, merge=True)
    RawDataset.run_preprocess(Convert("cloud", to="xyz"), seq)
    ds = RawDataset(seq, keys=["cloud", "cloud_xyz"])
    np.testing.assert_array_equal(ds.loaders["cloud_xyz"][1], ds.loaders["cloud"][1])


def test_a_filename_order_reaches_the_plugin_loader(tmp_path, plugins):
    """The core resolves ``order`` for a plugin as for a built-in: strays are
    filtered and unpadded indices sort numerically."""
    plugins(("xyz", "apairo_xyz:XyzFormat"))
    d = tmp_path / "seq" / "scan"
    d.mkdir(parents=True)
    for i in (1, 2, 10):
        (d / f"scan_{i}.xyz").write_text(f"{i} 0 0\n")
    (d / "notes.xyz.bak").write_text("not a frame\n")
    write_config(
        d.parent,
        {
            "version": 1,
            "channels": {
                "scan": {"kind": "raw", "loader": "xyz", "key": {"name": r"scan_(\d+)"}}
            },
        },
    )
    ds = RawDataset(d.parent)
    assert ds.loaders["scan"].files == ["scan_1.xyz", "scan_2.xyz", "scan_10.xyz"]
    np.testing.assert_allclose(ds.timestamps["scan"], [1.0, 2.0, 10.0])


# ───────────────── a stacked plugin with its own clock and field ──────────────


class JsonlLoader(AbstractLoader):
    """One JSON object per line; frame ``i`` is the member named by ``values``."""

    def __init__(self, directory, *, values: str) -> None:
        path = next(
            p for p in sorted(Path(directory).iterdir()) if p.suffix == ".jsonl"
        )
        self.records = [
            json.loads(line) for line in path.read_text().splitlines() if line
        ]
        self.values = values

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx):
        return np.asarray(self.records[idx][self.values], dtype=np.float64)

    @property
    def shape(self):
        return self[0].shape


class JsonlFormat(Format):
    name = "jsonl"
    loader = JsonlLoader
    extensions = frozenset({".jsonl"})
    per_frame = False
    fields = frozenset({"values"})
    key_forms = frozenset({"member"})

    def open(self, directory, meta, files=None):
        return JsonlLoader(directory, values=meta.get("values", "v"))

    def clock(self, loader, spec, label):
        return np.array([r[spec["member"]] for r in loader.records], dtype=float)

    def validate(self, key, meta, storage_dir):
        values = meta.get("values")
        if values is not None and not (isinstance(values, str) and values):
            return [f"channel '{key}': 'values' names a member, got {values!r}"]
        return []


@pytest.fixture
def jsonl_seq(tmp_path, plugins):
    plugins()  # no entry point: registered in process instead
    register_format(JsonlFormat)
    seq = tmp_path / "seq"
    (seq / "joints").mkdir(parents=True)
    lines = [json.dumps({"stamp": 5.0 + 0.01 * k, "q": [k, -k]}) for k in range(6)]
    (seq / "joints" / "arm.jsonl").write_text("\n".join(lines) + "\n")
    return seq


def test_a_plugin_brings_its_own_clock_form_and_field(jsonl_seq, capsys):
    seq = jsonl_seq
    RawDataset.init(seq)
    (seq / "apairo.yaml").write_text(
        "version: 1\nchannels:\n  joints:\n    values: q\n    key: {member: stamp}\n"
    )
    code, out = _cli(["check", str(seq)], capsys)
    assert code == 0, out  # 'values' is a known field, 'member' a known form

    _, out = _cli(["status", str(seq), "--json"], capsys)
    joints = json.loads(out)["channels"]["joints"]
    assert joints["frames"] == 6
    assert joints["shape"] == [2]
    assert joints["rate_hz"] == pytest.approx(100.0)

    ds = RawDataset(seq)
    np.testing.assert_allclose(ds.timestamps["joints"], 5.0 + 0.01 * np.arange(6))
    np.testing.assert_array_equal(ds.loaders["joints"][3], [3.0, -3.0])
    check_format(
        JsonlFormat(), seq / "joints", {"values": "q", "key": {"member": "stamp"}}
    )


def test_the_plugin_validates_its_fields_and_forms(jsonl_seq):
    seq = jsonl_seq
    (seq / "lidar").mkdir()
    for i in range(2):
        np.save(seq / "lidar" / f"{i:06d}.npy", np.zeros(3))
    channels = {
        "joints": {"kind": "raw", "loader": "jsonl", "values": 3},
        "lidar": {"kind": "raw", "loader": "npys", "key": {"member": "stamp"}},
    }
    write_config(seq, {"version": 1, "channels": channels})
    issues = verify_config(seq)
    assert any("'joints': 'values' names a member, got 3" in i for i in issues)
    assert any(
        "'lidar': 'key.member' reads the clock from the data and needs the "
        "'jsonl' loader (got 'npys')" in i
        for i in issues
    )

    channels["lidar"] = {"kind": "raw", "loader": "npys", "values": "q"}
    write_config(seq, {"version": 1, "channels": channels})
    assert any(
        "'values' is a format field and is only meaningful for the 'jsonl' loader "
        "(got 'npys')" in i
        for i in verify_config(seq)
    )


# ─────────────────────────────── the registry ─────────────────────────────────


def test_a_builtin_format_cannot_be_replaced(plugins):
    plugins()

    class Impostor(Format):
        name = "npys"
        loader = AbstractLoader

    with pytest.raises(ValueError, match="built-in format and cannot be replaced"):
        register_format(Impostor)


def test_a_taken_name_needs_replace(plugins):
    plugins(("xyz", "apairo_xyz:XyzFormat"))
    from apairo_xyz import XyzFormat

    get_format("xyz")
    with pytest.raises(ValueError, match="already registered"):
        register_format(XyzFormat)
    assert register_format(XyzFormat, replace=True).name == "xyz"


def test_a_format_needs_a_plain_name(plugins):
    plugins()

    class Spaced(Format):
        name = "two words"
        loader = AbstractLoader

    with pytest.raises(ValueError, match="plain `name`"):
        register_format(Spaced)


def test_a_broken_plugin_is_skipped_with_a_warning(plugins, caplog):
    plugins(
        ("broken", "apairo_missing_module:Nothing"), ("xyz", "apairo_xyz:XyzFormat")
    )
    with caplog.at_level(logging.WARNING, logger="apairo.core.formats"):
        names = registry.format_names()
    assert "xyz" in names and "broken" not in names
    assert "format plugin 'broken' was not loaded" in caplog.text


def test_an_unknown_loader_points_at_plugins(tmp_path, plugins, capsys):
    plugins()
    with pytest.raises(KeyError, match="Write a format plugin"):
        get_format("h5")
    seq = tmp_path / "seq"
    (seq / "scan").mkdir(parents=True)
    write_config(
        seq, {"version": 1, "channels": {"scan": {"kind": "raw", "loader": "h5"}}}
    )
    assert any(
        "unknown loader 'h5' (known: bin, csv, img, npy, npys, pcd, zarr)" in i
        for i in verify_config(seq)
    )


# ─────────────────────── the core never names a format ────────────────────────

# The modules on the read path of an asynchronous dataset: what a plugin must
# be able to reach without editing them. (The writers and the profiled
# datasets, whose profiles name their own files, are outside the contract.)
READ_PATH = [
    "apairo/cli.py",
    "apairo/core/config.py",
    "apairo/core/formats.py",
    "apairo/core/keys.py",
    "apairo/core/naming.py",
    "apairo/dataset/raw/dataset.py",
    *sorted(
        str(p.relative_to(REPO))
        for p in (REPO / "apairo/dataset/async_layout").glob("*.py")
    ),
]


@pytest.mark.parametrize("module", READ_PATH)
def test_the_read_path_never_names_a_builtin_format(module):
    tree = ast.parse((REPO / module).read_text(encoding="utf-8"))
    named = sorted(
        {
            f"line {node.lineno}: {node.value!r}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and node.value in BUILTINS
        }
    )
    assert named == [], (
        f"{module} names a format -- ask the registry (apairo.core.formats) instead"
    )
