"""Loader for a sequence-level table: one text file, one row per frame.

The shape of IMU logs, ground-truth trajectories and odometry exports in most
SLAM datasets -- EuRoC's ``imu0/data.csv``, TUM's ``groundtruth.txt`` -- where the
frame's clock is one of the columns rather than a filename or a sidecar.
"""

from __future__ import annotations

import re
from pathlib import Path

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
        names, rows = self._read(self.path)
        width = len(rows[0])

        self.key_tokens: list[str] | None = None
        key_idx: int | None = None
        if key_column is not None:
            key_idx = self._column_index(key_column, names, width, "key column")
            self.key_tokens = [row[key_idx] for row in rows]

        if fields is not None:
            if names is None:
                raise ValueError(
                    f"{self.path}: 'fields' selects columns by name, but the table "
                    f"has no header row or header comment to name them."
                )
            keep = [self._column_index(f, names, width, "field") for f in fields]
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
    def _read(path: Path) -> tuple[list[str] | None, list[list[str]]]:
        last_comment: str | None = None
        header: list[str] | None = None
        rows: list[list[str]] = []
        sep: str | None = None

        def split(line: str) -> list[str]:
            cells = line.split(sep) if sep is not None else line.split()
            return [c.strip() for c in cells]

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
                    continue
                if rows and len(cells) != len(rows[0]):
                    raise ValueError(
                        f"{path}:{lineno}: {len(cells)} column(s), expected "
                        f"{len(rows[0])}."
                    )
                rows.append(cells)

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
        return header, rows

    def _column_index(
        self, column: int | str, names: list[str] | None, width: int, what: str
    ) -> int:
        if isinstance(column, bool) or not isinstance(column, (int, str)):
            raise ValueError(
                f"{self.path}: {what} {column!r} is not an index or a name."
            )
        if isinstance(column, int):
            if not 0 <= column < width:
                raise ValueError(
                    f"{self.path}: {what} {column} is out of range for {width} "
                    f"column(s)."
                )
            return column
        if names is None or column not in names:
            known = f"; columns: {names}" if names is not None else " (no header)"
            raise ValueError(f"{self.path}: no column named {column!r}{known}.")
        return names.index(column)

    # ------------------------------------------------------------------ access

    def __len__(self) -> int:
        return len(self.array)

    def __getitem__(self, idx: int) -> np.ndarray:
        # Copy, like NPYLoader: the table is a persistent whole-array cache.
        return self.array[idx].copy()

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.array.shape[1:])
