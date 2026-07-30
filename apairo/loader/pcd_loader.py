"""Loader for PCL ``.pcd`` point clouds (v0.7, ``ascii`` and ``binary``).

A PCD file is self-describing: its header names the fields it carries, so two
files in the same directory may legitimately hold different ones. That is the
whole reason this loader exists -- ``bin`` assumes a headerless float32 quadruple
and would read the ASCII header as coordinates.

The loader stays pure format mechanics: it parses a header and returns numbers.
*Which* fields a channel is contractually expected to expose is the layout's
business, declared as ``fields:`` in ``channels.yaml`` and passed in here. This
loader never invents a projection of its own -- without ``fields`` it returns
every field the header declares, in the header's own order.
"""

from __future__ import annotations

import os

import numpy as np

from apairo.core import AbstractLoader

# PCD TYPE/SIZE pairs -> numpy scalar types. The header's TYPE is one of
# I (signed), U (unsigned), F (float); SIZE is the width in bytes.
_TYPE_TO_NP: dict[tuple[str, int], np.dtype] = {
    ("I", 1): np.dtype(np.int8),
    ("I", 2): np.dtype(np.int16),
    ("I", 4): np.dtype(np.int32),
    ("I", 8): np.dtype(np.int64),
    ("U", 1): np.dtype(np.uint8),
    ("U", 2): np.dtype(np.uint16),
    ("U", 4): np.dtype(np.uint32),
    ("U", 8): np.dtype(np.uint64),
    ("F", 4): np.dtype(np.float32),
    ("F", 8): np.dtype(np.float64),
}

_HEADER_KEYS = frozenset(
    {
        "VERSION",
        "FIELDS",
        "SIZE",
        "TYPE",
        "COUNT",
        "WIDTH",
        "HEIGHT",
        "VIEWPOINT",
        "POINTS",
        "DATA",
    }
)


class PCDHeader:
    """A parsed PCD header: the columns it declares and where the payload starts.

    ``columns`` is the logical field list -- one entry per scalar column, in the
    file's own order, as ``(name, numpy dtype)``. A field with ``COUNT > 1``
    expands to ``<name>_0 .. <name>_<count-1>``. PCL's ``_`` padding fields are
    dropped from it, but they still occupy bytes, so they stay in ``record`` (the
    binary point stride) and ``flat_index`` maps a logical column back to its
    position among *all* scalar columns -- the order both encodings write.
    """

    def __init__(
        self,
        fields: list[str],
        sizes: list[int],
        types: list[str],
        counts: list[int],
        width: int,
        height: int,
        points: int | None,
        data: str,
        end: int,
    ) -> None:
        self.data_format: str = data
        self.n_points: int = points if points is not None else width * height
        self.width, self.height = width, height
        self.data_offset: int = end

        columns: list[tuple[str, np.dtype]] = []
        record: list[tuple[str, np.dtype, int]] = []
        flat_index: list[int] = []
        at = 0
        for i, (name, size, typ, count) in enumerate(
            zip(fields, sizes, types, counts, strict=True)
        ):
            dtype = _TYPE_TO_NP.get((typ, size))
            if dtype is None:
                raise ValueError(
                    f"Unsupported PCD field '{name}': TYPE {typ} SIZE {size}. "
                    f"Known combinations: {sorted(_TYPE_TO_NP)}."
                )
            if count < 1:
                raise ValueError(f"PCD field '{name}' has COUNT {count}.")
            # Unique storage name: PCL emits '_' for every padding field, and a
            # numpy structured dtype needs distinct names.
            record.append((f"f{i}", dtype, count))
            if name != "_":
                names = [name] if count == 1 else [f"{name}_{j}" for j in range(count)]
                columns += [(n, dtype) for n in names]
                flat_index += list(range(at, at + count))
            at += count
        self.columns = columns
        self.record = record
        self.flat_index = flat_index
        self.n_scalars = at

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.columns)

    def select(self, fields: list[str] | None) -> list[int]:
        """Logical column indices for *fields*, or all of them when None."""
        if fields is None:
            return list(range(len(self.columns)))
        index = {name: i for i, (name, _) in enumerate(self.columns)}
        missing = [f for f in fields if f not in index]
        if missing:
            raise ValueError(
                f"PCD field(s) {missing} missing. Channel declares "
                f"{list(fields)}; the file carries {list(self.names)}."
            )
        return [index[f] for f in fields]


def _parse_header(path: str) -> PCDHeader:
    """Read *path*'s header. Stops at the DATA line -- the payload is untouched."""
    entries: dict[str, list[str]] = {}
    with open(path, "rb") as fh:
        while True:
            raw = fh.readline()
            if not raw:
                raise ValueError(f"Truncated PCD header (no DATA line): '{path}'.")
            line = raw.decode("ascii", "replace").strip()
            if not line or line.startswith("#"):
                continue
            token, _, rest = line.partition(" ")
            token = token.upper()
            if token not in _HEADER_KEYS:
                continue  # tolerate a line a newer PCD revision adds
            entries[token] = rest.split()
            if token == "DATA":
                offset = fh.tell()
                break

    for required in ("FIELDS", "SIZE", "TYPE", "DATA"):
        if required not in entries:
            raise ValueError(f"PCD header of '{path}' has no {required} line.")

    fields = entries["FIELDS"]
    sizes = [int(s) for s in entries["SIZE"]]
    types = [t.upper() for t in entries["TYPE"]]
    counts = [int(c) for c in entries.get("COUNT", ["1"] * len(fields))]
    if not (len(fields) == len(sizes) == len(types) == len(counts)):
        raise ValueError(
            f"Inconsistent PCD header in '{path}': FIELDS/SIZE/TYPE/COUNT have "
            f"{len(fields)}/{len(sizes)}/{len(types)}/{len(counts)} entries."
        )

    data_format = entries["DATA"][0].lower() if entries["DATA"] else ""
    if data_format == "binary_compressed":
        raise ValueError(
            f"'{path}' is DATA binary_compressed, which apairo does not decode "
            f"(it needs an LZF decompressor). Re-save it uncompressed, e.g. "
            f"`pcl_convert_pcd_ascii_binary in.pcd out.pcd 1`."
        )
    if data_format not in ("ascii", "binary"):
        raise ValueError(
            f"Unknown PCD DATA format '{data_format}' in '{path}' "
            f"(expected ascii or binary)."
        )

    return PCDHeader(
        fields=fields,
        sizes=sizes,
        types=types,
        counts=counts,
        width=int(entries.get("WIDTH", [0])[0]),
        height=int(entries.get("HEIGHT", [1])[0]),
        points=int(entries["POINTS"][0]) if "POINTS" in entries else None,
        data=data_format,
        end=offset,
    )


def _read_ascii(path: str, header: PCDHeader) -> np.ndarray:
    """The ascii payload as an ``(N, n_scalars)`` float64 table.

    float64 is the parse type, not the result type: it holds every PCD scalar
    except a full-range 64-bit integer exactly, and the caller casts down to the
    field's own type."""
    with open(path, "rb") as fh:
        fh.seek(header.data_offset)
        table = np.loadtxt(fh, dtype=np.float64, ndmin=2, comments="#")
    if table.size == 0:
        return np.empty((0, header.n_scalars), dtype=np.float64)
    if table.shape[1] != header.n_scalars:
        raise ValueError(
            f"PCD ascii payload of '{path}' has {table.shape[1]} column(s) but "
            f"its header declares {header.n_scalars}."
        )
    return table


def _read_binary(path: str, header: PCDHeader) -> list[np.ndarray]:
    """The binary payload as one array per scalar column, padding included."""
    record = np.dtype([(n, d, c) if c > 1 else (n, d) for n, d, c in header.record])
    want = record.itemsize * header.n_points
    with open(path, "rb") as fh:
        fh.seek(header.data_offset)
        buf = fh.read(want)
    if len(buf) < want:
        raise ValueError(
            f"Truncated PCD payload in '{path}': {len(buf)} byte(s) for "
            f"{header.n_points} point(s) of {record.itemsize} byte(s)."
        )
    flat = np.frombuffer(buf, dtype=record)
    cols: list[np.ndarray] = []
    for name in record.names or ():
        col = flat[name]
        cols += [col] if col.ndim == 1 else [col[:, j] for j in range(col.shape[1])]
    return cols


def read_pcd(path: str, fields: list[str] | None = None) -> np.ndarray:
    """One ``.pcd`` file as an ``(N, C)`` array, restricted to *fields* if given.

    The result dtype is the smallest numpy type that holds every selected field
    exactly (``np.result_type`` over their native dtypes), so an all-``F4`` cloud
    stays float32 while an Ouster ``t`` / ``range`` (``U4``) promotes to float64
    rather than silently losing counts to a float32 mantissa.
    """
    header = _parse_header(path)
    picked = header.select(fields)
    if not picked:
        return np.empty((header.n_points, 0), dtype=np.float32)
    out_dtype = np.result_type(*[header.columns[i][1] for i in picked])
    scalar_cols = [header.flat_index[i] for i in picked]

    if header.data_format == "ascii":
        table = _read_ascii(path, header)
        return np.ascontiguousarray(table[:, scalar_cols].astype(out_dtype))

    cols = _read_binary(path, header)
    out = np.empty((header.n_points, len(picked)), dtype=out_dtype)
    for i, src in enumerate(scalar_cols):
        out[:, i] = cols[src]
    return out


class PCDLoader(AbstractLoader):
    """Loader for a directory of per-frame PCL ``.pcd`` clouds.

    Like every apairo loader this is pure mechanics: it imposes no naming
    convention (the dataset passes ``files`` in frame order) and no field
    convention (the layout passes ``fields``).

    Args:
        directory: Directory holding the ``.pcd`` frames.
        files: Frame-ordered file names resolved by the dataset. Defaults to
            every ``.pcd`` in *directory*, sorted.
        fields: The channel's field contract, declared in ``channels.yaml``.
            Selects those columns, in that order, so the returned width is stable
            across files whose headers differ; a file missing one of them raises,
            naming both field sets. ``None`` returns every field the file
            declares, in header order -- convenient, but then the width is only
            as stable as the files are.
    """

    def __init__(
        self,
        directory: str,
        files: list[str] | None = None,
        fields: list[str] | None = None,
    ) -> None:
        if files is not None:
            self.files = list(files)
        else:
            self.files = sorted(
                f for f in os.listdir(directory) if f.lower().endswith(".pcd")
            )
        if not self.files:
            raise FileNotFoundError(f"No .pcd frames found in {directory}")
        self.directory = directory
        self.fields = list(fields) if fields is not None else None
        first = _parse_header(os.path.join(directory, self.files[0]))
        self._field_names = first.names
        self._shape = (len(self.fields) if self.fields else len(first.columns),)

    @property
    def field_names(self) -> tuple[str, ...]:
        """The first frame's fields, in the header's own order. Descriptive: a
        later frame may declare others -- that is what ``fields`` guards."""
        return self._field_names

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx) -> np.ndarray:
        # One file read per frame, header included: the header is the only thing
        # that says which fields *this* file carries.
        return read_pcd(os.path.join(self.directory, self.files[idx]), self.fields)
