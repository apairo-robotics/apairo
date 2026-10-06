"""Loader for a sequence-level table: one text file, one row per frame.

The shape of IMU logs, ground-truth trajectories and odometry exports in most
SLAM datasets -- EuRoC's ``imu0/data.csv``, TUM's ``groundtruth.txt`` -- where the
frame's clock is one of the columns rather than a filename or a sidecar.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import numpy as np

from apairo.core import AbstractLoader
from apairo.core.utils.exceptions import FileExtensionError

# A header cell's trailing unit annotation, as in EuRoC's "w_RS_S_x [rad s^-1]".
_UNIT_SUFFIX = re.compile(r"\s*\[[^\]]*\]\s*$")


def _column_name(cell: str) -> str:
    """A header cell as a column name: a leading ``#`` and a trailing ``[unit]``
    annotation dropped, surrounding blanks stripped."""
    return _UNIT_SUFFIX.sub("", cell.strip().lstrip("#").strip())


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


# A .txt file is only taken for a table when its first rows read as one: many are
# notes, index files (TUM's rgb.txt pairs a stamp with a filename) or sidecars.
_TABLE_SNIFF_ROWS = 20


def looks_like_table(path: Path) -> bool:
    """True when *path* starts like a numeric table the ``csv`` loader reads:
    ``#`` comments, at most one header row, then rows of numbers of one width.
    Only the first rows are read."""
    width: int | None = None
    header_seen = False
    rows = 0
    try:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                sep = "," if "," in line else ("\t" if "\t" in line else None)
                cells = [c.strip() for c in (line.split(sep) if sep else line.split())]
                try:
                    [float(c) for c in cells]
                except ValueError:
                    if width is not None or header_seen:
                        return False  # text after the first data row: not a table
                    header_seen = True
                    continue
                if width is not None and len(cells) != width:
                    return False
                width = len(cells)
                rows += 1
                if rows >= _TABLE_SNIFF_ROWS:
                    break
    except (OSError, UnicodeDecodeError):
        return False
    return width is not None


class _Head(NamedTuple):
    names: list[str] | None
    rows: list[list[str]]
    sep: str | None
    header_row: bool


class TableScan(NamedTuple):
    """What :func:`scan_table` reads off a table without parsing its cells."""

    rows: int
    columns: list[str] | None
    width: int
    first_key: str | None
    last_key: str | None


def scan_table(
    directory: str | Path,
    *,
    file: str | None = None,
    key_column: int | str | None = None,
    fields: list[str] | None = None,
) -> TableScan:
    """The size and clock span of the table :class:`CSVLoader` would read, for
    ``apairo status``: names and the first row from the head, then one pass
    over the lines -- counting the rows, checking their width, keeping the
    last -- with no cell converted to a number. The loader's rules apply
    (comments, header row or header comment, ``key_column``, ``fields``); a
    non-numeric cell is left to loading to report."""
    path = CSVLoader._locate(Path(directory), file)
    head = CSVLoader._read(path, limit=1)
    width = len(head.rows[0])
    sep = head.sep.encode() if head.sep is not None else None
    data_lines = 0
    last = b""
    with open(path, "rb") as f:
        for lineno, raw in enumerate(f, start=1):
            first = raw[:1]
            if first == b"#" or raw.isspace():
                continue
            if first in b" \t" and raw.lstrip()[:1] == b"#":
                continue
            data_lines += 1
            if data_lines == 1 and head.header_row:
                continue
            cells = raw.count(sep) + 1 if sep is not None else len(raw.split())
            if cells != width:
                raise ValueError(
                    f"{path}:{lineno}: {cells} column(s), expected {width}."
                )
            last = raw
    rows = data_lines - head.header_row
    key_idx = (
        _column_index(path, key_column, head.names, width, "key column")
        if key_column is not None
        else None
    )
    if fields is not None:
        if head.names is None:
            raise ValueError(
                f"{path}: 'fields' selects columns by name, but the table has no "
                f"header row or header comment to name them."
            )
        keep = [_column_index(path, f, head.names, width, "field") for f in fields]
    else:
        keep = [i for i in range(width) if i != key_idx]
    last_cells = _split(last.decode("utf-8").strip(), head.sep)
    return TableScan(
        rows=rows,
        columns=[head.names[i] for i in keep] if head.names is not None else None,
        width=len(keep),
        first_key=head.rows[0][key_idx] if key_idx is not None else None,
        last_key=last_cells[key_idx] if key_idx is not None else None,
    )


def _split(line: str, sep: str | None) -> list[str]:
    cells = line.split(sep) if sep is not None else line.split()
    return [c.strip() for c in cells]


def _column_index(
    path: Path, column: int | str, names: list[str] | None, width: int, what: str
) -> int:
    if isinstance(column, bool) or not isinstance(column, (int, str)):
        raise ValueError(f"{path}: {what} {column!r} is not an index or a name.")
    if isinstance(column, int):
        if not 0 <= column < width:
            raise ValueError(
                f"{path}: {what} {column} is out of range for {width} column(s)."
            )
        return column
    if names is None or column not in names:
        known = f"; columns: {names}" if names is not None else " (no header)"
        raise ValueError(f"{path}: no column named {column!r}{known}.")
    return names.index(column)


class CSVLoader(AbstractLoader):
    r"""Loader for a delimited text table in a channel directory, one row per frame.

    The file is the directory's single ``.csv`` (or the one named by ``file`` --
    the channel's ``array_file``, needed when the directory colocates several, or
    for a ``.txt`` table). Rows are comma-, tab- or whitespace-separated, the
    separator being read off the first data row. Lines starting with ``#`` are
    comments; column names come from a non-numeric first row, else from the last
    comment line when it splits into exactly one name per column (EuRoC's
    ``#timestamp [ns],w_RS_S_x [rad s^-1],...``, TUM's ``# timestamp tx ty ...``).
    Names drop the leading ``#`` and a trailing ``[unit]``.

    Args:
        directory: The channel directory.
        file: The table's filename inside *directory*.
        key_column: The column holding each row's clock (an index, or a column
            name). It is kept out of the frame data and exposed, untouched, as
            :attr:`key_tokens` for the channel's ``key: {column: ...}`` spec.
        fields: Data columns to keep, by name and in this order -- the channel's
            declared width. Without it, every column but the key column.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        file: str | None = None,
        key_column: int | str | None = None,
        fields: list[str] | None = None,
    ) -> None:
        self.path = self._locate(Path(directory), file)
        names, rows, _, _ = self._read(self.path)
        width = len(rows[0])

        self.key_tokens: list[str] | None = None
        key_idx: int | None = None
        if key_column is not None:
            key_idx = _column_index(self.path, key_column, names, width, "key column")
            self.key_tokens = [row[key_idx] for row in rows]

        if fields is not None:
            if names is None:
                raise ValueError(
                    f"{self.path}: 'fields' selects columns by name, but the table "
                    f"has no header row or header comment to name them."
                )
            keep = [_column_index(self.path, f, names, width, "field") for f in fields]
        else:
            keep = [i for i in range(width) if i != key_idx]
        self.columns: list[str] | None = (
            [names[i] for i in keep] if names is not None else None
        )

        try:
            self.array: np.ndarray = np.array(
                [[float(row[i]) for i in keep] for row in rows], dtype=np.float64
            ).reshape(len(rows), len(keep))
        except ValueError as exc:
            raise ValueError(f"{self.path}: non-numeric data cell ({exc}).") from exc

    # ----------------------------------------------------------------- parsing

    @staticmethod
    def _locate(directory: Path, file: str | None) -> Path:
        if file is not None:
            target = directory / file
            if not target.is_file():
                raise FileExtensionError(f"No such table file: {target}")
            return target
        tables = sorted(directory.glob("*.csv"))
        if not tables:
            raise FileExtensionError(
                f"No .csv file found in {directory}; name a .txt table with "
                f"'array_file'."
            )
        if len(tables) > 1:
            raise FileExtensionError(
                f"Several .csv files in {directory} "
                f"({', '.join(t.name for t in tables)}); name one with 'array_file'."
            )
        return tables[0]

    @staticmethod
    def _read(path: Path, limit: int | None = None) -> _Head:
        """Names, data rows (the first *limit* only, when given), separator, and
        whether the names came from a header row rather than a comment."""
        last_comment: str | None = None
        header: list[str] | None = None
        header_row = False
        rows: list[list[str]] = []
        sep: str | None = None

        def split(line: str) -> list[str]:
            return _split(line, sep)

        with open(path, encoding="utf-8") as f:
            for lineno, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    if not rows and header is None:
                        last_comment = line
                    continue
                if sep is None and not rows and header is None:
                    sep = "," if "," in line else ("\t" if "\t" in line else None)
                cells = split(line)
                if not rows and header is None and not all(map(_is_number, cells)):
                    header = [_column_name(c) for c in cells]
                    header_row = True
                    continue
                if rows and len(cells) != len(rows[0]):
                    raise ValueError(
                        f"{path}:{lineno}: {len(cells)} column(s), expected "
                        f"{len(rows[0])}."
                    )
                rows.append(cells)
                if limit is not None and len(rows) >= limit:
                    break

        if not rows:
            raise ValueError(f"{path}: no data rows.")
        if header is None and last_comment is not None:
            candidate = [_column_name(c) for c in split(last_comment.lstrip("#"))]
            if len(candidate) == len(rows[0]) and all(candidate):
                header = candidate
        if header is not None and len(header) != len(rows[0]):
            raise ValueError(
                f"{path}: header has {len(header)} name(s) for {len(rows[0])} "
                f"column(s)."
            )
        return _Head(header, rows, sep, header_row)

    # ------------------------------------------------------------------ access

    def __len__(self) -> int:
        return len(self.array)

    def __getitem__(self, idx: int) -> np.ndarray:
        # Copy, like NPYLoader: the table is a persistent whole-array cache.
        return self.array[idx].copy()

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.array.shape[1:])
