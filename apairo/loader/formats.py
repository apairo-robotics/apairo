"""The built-in formats, registered on import through the same contract a
plugin uses (:mod:`apairo.core.formats`). Nothing here is special-cased in the
core: every format-specific decision lives in one of these classes."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from apairo.core.formats import Facts, Format, register_format
from apairo.core.keys import epoch_unit, parse_column_key
from apairo.core.naming import is_frame_file
from apairo.loader.bin_loader import BINLoader
from apairo.loader.csv_loader import CSVLoader, looks_like_table
from apairo.loader.img_loader import IMGLoader
from apairo.loader.npy_loader import NPYLoader
from apairo.loader.npys_loader import NPYSLoader
from apairo.loader.pcd_loader import PCDLoader, _parse_header
from apairo.loader.zarr_loader import ZarrLoader


class NpysFormat(Format):
    """One ``.npy`` per frame (``000000.npy``), with optional suffixed variants
    (``000000_intensity.npy``) that fan out into sibling channels."""

    name = "npys"
    loader = NPYSLoader
    extensions = frozenset({".npy"})
    suffixes = True
    priority = 50

    def detect(self, directory: Path) -> bool:
        # Several .npy files are frames; a single one is a stacked array (npy).
        return len(self.data_files(directory)) > 1

    def facts(
        self, directory: Path, meta: dict, files: list[str] | None = None
    ) -> Facts:
        if files is None:  # the loader's own listing: no suffixed variant
            files = sorted(
                f.name for f in self.data_files(directory) if is_frame_file(f.name)
            )
        if not files:
            return Facts(frames=0)
        arr = np.load(directory / files[0], mmap_mode="r")  # header only
        return Facts(frames=len(files), shape=list(arr.shape), dtype=str(arr.dtype))


class NpyFormat(Format):
    """One stacked ``.npy`` array, row ``i`` being frame ``i``. ``array_file``
    names it when the directory colocates several."""

    name = "npy"
    loader = NPYLoader
    extensions = frozenset({".npy"})
    per_frame = False
    fields = frozenset({"array_file"})
    priority = 51

    def detect(self, directory: Path) -> bool:
        return len(self.data_files(directory)) == 1

    def open(self, directory: Path, meta: dict, files: list[str] | None = None):
        return NPYLoader(directory, file=meta.get("array_file"))

    def facts(
        self, directory: Path, meta: dict, files: list[str] | None = None
    ) -> Facts:
        if meta.get("array_file"):
            target: Path | None = directory / str(meta["array_file"])
        else:
            target = next(iter(self.data_files(directory)), None)
        if target is None or not target.is_file():
            return Facts()
        arr = np.load(target, mmap_mode="r")  # header only
        return Facts(frames=len(arr), shape=list(arr.shape[1:]), dtype=str(arr.dtype))


class BinFormat(Format):
    """One headerless float32 ``.bin`` per frame, ``(x, y, z, intensity)``."""

    name = "bin"
    loader = BINLoader
    extensions = frozenset({".bin"})
    priority = 20

    def facts(
        self, directory: Path, meta: dict, files: list[str] | None = None
    ) -> Facts:
        if files is None:
            files = [f.name for f in self.data_files(directory)]
        if not files:
            return Facts(frames=0)
        size = (directory / files[0]).stat().st_size  # sized, not read
        return Facts(frames=len(files), shape=[size // 16, 4], dtype="float32")


class PcdFormat(Format):
    """One PCL ``.pcd`` point cloud per frame; ``fields`` pins the columns kept,
    in order, since a PCD header is self-describing per file."""

    name = "pcd"
    loader = PCDLoader
    extensions = frozenset({".pcd"})
    fields = frozenset({"fields"})
    priority = 30

    def open(self, directory: Path, meta: dict, files: list[str] | None = None):
        fields = list(meta["fields"]) if meta.get("fields") else None
        return PCDLoader(str(directory), files=files, fields=fields)

    def declare_hints(self, directory: Path, meta: dict) -> list[str]:
        first = next(iter(self.data_files(directory)), None)
        if first is None:
            return []
        try:
            names = ", ".join(_parse_header(str(first)).names)
        except (OSError, ValueError):
            return []
        return [f"    # fields: [{names}]   # keep the columns you need, in this order"]


class ImgFormat(Format):
    """One image per frame (PNG, JPEG, BMP), decoded with Pillow."""

    name = "img"
    loader = IMGLoader
    extensions = frozenset({".png", ".jpg", ".jpeg", ".bmp"})
    priority = 40


class ZarrFormat(Format):
    """A Zarr array store as the channel directory itself, row ``i`` being frame
    ``i``, with its ``timestamps.txt`` beside the chunks."""

    name = "zarr"
    loader = ZarrLoader
    per_frame = False
    priority = 10

    def detect(self, directory: Path) -> bool:
        return (directory / ".zarray").exists() or (directory / "zarr.json").exists()

    def open(self, directory: Path, meta: dict, files: list[str] | None = None):
        return ZarrLoader(directory)


# Header names (after the csv loader drops '#' and '[unit]') that name a clock.
_CLOCK_COLUMNS = frozenset(
    {"t", "time", "times", "timestamp", "timestamps", "stamp", "stamps", "ts"}
)


class CsvFormat(Format):
    """A delimited text table, one row per frame, its clock in a column
    (``key: {column: ...}``). Each table is a channel of its own."""

    name = "csv"
    loader = CSVLoader
    extensions = frozenset({".csv", ".txt"})
    per_frame = False
    one_channel_per_file = True
    fields = frozenset({"array_file", "fields"})
    key_forms = frozenset({"column"})
    priority = 90

    def matches(self, path: Path) -> bool:
        # A .txt or .csv is a table only when its first rows read as one: an
        # index pairing stamps with filenames (EuRoC's cam0/data.csv, TUM's
        # rgb.txt) or a note is not data.
        return path.suffix.lower() in self.extensions and looks_like_table(path)

    def detect(self, directory: Path) -> bool:
        # A channel directory is a table only through a .csv: a stray .txt beside
        # other files is more often a note than the channel's data.
        return any(p.suffix.lower() == ".csv" for p in self.data_files(directory))

    def open(self, directory: Path, meta: dict, files: list[str] | None = None):
        key = meta.get("key")
        spec: dict = key if isinstance(key, dict) else {}
        fields = list(meta["fields"]) if meta.get("fields") else None
        return CSVLoader(
            directory,
            file=meta.get("array_file"),
            key_column=spec.get("column"),
            fields=fields,
        )

    def clock(self, loader, spec: dict, label: str) -> np.ndarray:
        tokens = getattr(loader, "key_tokens", None)
        if tokens is None:
            raise ValueError(f"{label}: the table was opened without its key column.")
        return parse_column_key(tokens, spec, label=label)

    def facts(
        self, directory: Path, meta: dict, files: list[str] | None = None
    ) -> Facts:
        table = self.open(directory, meta)
        key = meta.get("key")
        spec: dict = key if isinstance(key, dict) else {}
        clock = (
            parse_column_key(table.key_tokens, spec)
            if table.key_tokens is not None
            else None
        )
        return Facts(
            frames=len(table),
            shape=list(table.shape),
            dtype=str(table.array.dtype),
            clock=clock,
        )

    def declare_hints(self, directory: Path, meta: dict) -> list[str]:
        """A ``key: {column: ...}`` line when the table has no clock on disk: the
        column whose header names a clock and never decreases, its unit guessed
        from the width of an epoch -- uncommented when both hold, so the
        scaffold loads as generated."""
        if "array_file" in meta:
            table: Path | None = directory / str(meta["array_file"])
        else:
            if (directory / "timestamps.txt").exists():
                return []
            table = next(
                iter(p for p in self.data_files(directory) if p.suffix == ".csv"), None
            )
        if table is None:
            return []
        homework = (
            "    # key: {column: <index or name>, units: [s]}"
            "   # no timestamps.txt -- name the clock column"
        )
        try:
            data = CSVLoader(table.parent, file=table.name)
        except Exception:
            return [homework]
        names = data.columns or []
        candidates = [i for i, n in enumerate(names) if n.lower() in _CLOCK_COLUMNS]
        if not candidates:
            return [homework]
        i = candidates[0]
        values = data.array[:1000, i]
        if values.size == 0 or (values.size > 1 and np.any(np.diff(values) < 0)):
            return [homework]
        digits, unit = epoch_unit(float(values[0]))
        column = (
            names[i]
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", names[i])
            else f"'{names[i]}'"
        )
        if unit is None:
            return [
                f"    key: {{column: {column}}}"
                f"   # clock column '{names[i]}', seconds assumed -- verify"
            ]
        return [
            f"    key: {{column: {column}, units: [{unit}]}}"
            f"   # {digits}-digit epoch in column '{names[i]}' -- verify"
        ]

    def validate(self, key: str, meta: dict, storage_dir: Path) -> list[str]:
        spec = meta.get("key")
        if not isinstance(spec, dict) or "column" not in spec:
            return []
        from apairo.core.config import KEY_UNITS

        out: list[str] = []
        column = spec["column"]
        if isinstance(column, bool) or not (
            (isinstance(column, int) and column >= 0)
            or (isinstance(column, str) and column)
        ):
            out.append(
                f"channel '{key}': 'key.column' must be a column index "
                f"(>= 0) or a column name, got {column!r}"
            )
        units = spec.get("units", spec.get("scale"))
        if spec.get("units") is not None and spec.get("scale") is not None:
            out.append(
                f"channel '{key}': 'key' has both 'units' and 'scale' -- "
                f"'units' is sugar for 'scale', give one"
            )
        elif units is not None and not (isinstance(units, list) and len(units) == 1):
            out.append(
                f"channel '{key}': a column key takes a one-entry "
                f"'units'/'scale' list, got {units!r}"
            )
        elif spec.get("units") is not None and spec["units"][0] not in KEY_UNITS:
            out.append(
                f"channel '{key}': 'key.units' has unknown unit(s) "
                f"{spec['units']}; known: {sorted(KEY_UNITS)}"
            )
        return out


for _fmt in (
    NpysFormat,
    NpyFormat,
    BinFormat,
    PcdFormat,
    ImgFormat,
    ZarrFormat,
    CsvFormat,
):
    register_format(_fmt)
