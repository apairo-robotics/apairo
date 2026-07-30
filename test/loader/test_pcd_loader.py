"""PCD loader: header parsing, both payload encodings, and the field contract."""

import numpy as np
import pytest

from apairo.loader import DERIVED_LOADERS, str_to_loader
from apairo.loader.pcd_loader import PCDLoader, _parse_header, read_pcd

XYZI = [("x", "F", 4), ("y", "F", 4), ("z", "F", 4), ("intensity", "F", 4)]


def _header(fields, n_points, data, *, counts=None, height=1) -> str:
    counts = counts or [1] * len(fields)
    return (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        f"FIELDS {' '.join(f[0] for f in fields)}\n"
        f"SIZE {' '.join(str(f[2]) for f in fields)}\n"
        f"TYPE {' '.join(f[1] for f in fields)}\n"
        f"COUNT {' '.join(str(c) for c in counts)}\n"
        f"WIDTH {n_points // height}\n"
        f"HEIGHT {height}\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n_points}\n"
        f"DATA {data}\n"
    )


def write_ascii(path, fields, rows, *, counts=None) -> None:
    body = "\n".join(" ".join(repr(v) for v in row) for row in rows)
    path.write_text(_header(fields, len(rows), "ascii", counts=counts) + body + "\n")


def write_binary(path, fields, rows, *, counts=None) -> None:
    counts = counts or [1] * len(fields)
    np_types = {("F", 4): np.float32, ("F", 8): np.float64, ("U", 4): np.uint32}
    dtype = np.dtype(
        [
            (f[0], np_types[(f[1], f[2])], c)
            if c > 1
            else (f[0], np_types[(f[1], f[2])])
            for f, c in zip(fields, counts, strict=True)
        ]
    )
    rec = np.zeros(len(rows), dtype=dtype)
    for i, row in enumerate(rows):
        at = 0
        for (name, _, _), c in zip(fields, counts, strict=True):
            rec[name][i] = row[at] if c == 1 else row[at : at + c]
            at += c
    with open(path, "wb") as fh:
        fh.write(_header(fields, len(rows), "binary", counts=counts).encode())
        fh.write(rec.tobytes())


ROWS = [[1.0, 2.0, 3.0, 10.0], [4.0, 5.0, 6.0, 20.0]]


@pytest.fixture(params=["ascii", "binary"])
def cloud(request, tmp_path):
    """The same two points, written in each payload encoding."""
    path = tmp_path / "000000.pcd"
    writer = write_ascii if request.param == "ascii" else write_binary
    writer(path, XYZI, ROWS)
    return path


# ---------------------------------------------------------------- payloads


def test_reads_both_encodings(cloud):
    np.testing.assert_allclose(read_pcd(str(cloud)), np.array(ROWS, dtype=np.float32))


def test_all_float32_fields_stay_float32(cloud):
    assert read_pcd(str(cloud)).dtype == np.float32


def test_projects_declared_fields_in_order(cloud):
    out = read_pcd(str(cloud), ["intensity", "x"])
    np.testing.assert_allclose(out, np.array([[10.0, 1.0], [20.0, 4.0]]))


def test_missing_field_names_both_sets(cloud):
    with pytest.raises(ValueError, match="rgb"):
        read_pcd(str(cloud), ["x", "y", "z", "rgb"])
    with pytest.raises(ValueError, match="intensity"):  # the file's own fields
        read_pcd(str(cloud), ["rgb"])


def test_count_gt_one_expands_to_indexed_columns(tmp_path):
    fields = [("x", "F", 4), ("normal", "F", 4)]
    path = tmp_path / "n.pcd"
    write_binary(path, fields, [[1.0, 7.0, 8.0, 9.0]], counts=[1, 3])
    assert _parse_header(str(path)).names == ("x", "normal_0", "normal_1", "normal_2")
    np.testing.assert_allclose(read_pcd(str(path), ["normal_1"]), [[8.0]])


@pytest.mark.parametrize("writer", [write_ascii, write_binary])
def test_padding_fields_are_dropped_but_still_consume_their_slot(writer, tmp_path):
    """PCL emits `_` padding fields; they are not data, but they shift columns."""
    fields = [("x", "F", 4), ("_", "F", 4), ("intensity", "F", 4)]
    path = tmp_path / "p.pcd"
    writer(path, fields, [[1.0, 999.0, 42.0]])
    assert _parse_header(str(path)).names == ("x", "intensity")
    np.testing.assert_allclose(read_pcd(str(path)), [[1.0, 42.0]])


@pytest.mark.parametrize("writer", [write_ascii, write_binary])
def test_uint32_field_promotes_to_float64_rather_than_losing_counts(writer, tmp_path):
    """An Ouster `t` is U4; float32 has a 24-bit mantissa and would round it."""
    fields = [("x", "F", 4), ("t", "U", 4)]
    stamp = 4_000_000_001  # not representable in float32
    path = tmp_path / "t.pcd"
    writer(path, fields, [[1.0, stamp]])
    out = read_pcd(str(path))
    assert out.dtype == np.float64
    assert int(out[0, 1]) == stamp


def test_organized_cloud_flattens_to_width_times_height(tmp_path):
    path = tmp_path / "org.pcd"
    write_binary(path, XYZI, ROWS)  # POINTS says 2
    text = path.read_bytes().replace(b"WIDTH 2\nHEIGHT 1", b"WIDTH 1\nHEIGHT 2")
    path.write_bytes(text)
    assert read_pcd(str(path)).shape == (2, 4)


# ---------------------------------------------------------------- rejections


def test_binary_compressed_is_refused_by_name(tmp_path):
    path = tmp_path / "c.pcd"
    path.write_bytes(_header(XYZI, 1, "binary_compressed").encode() + b"\x00" * 16)
    with pytest.raises(ValueError, match="binary_compressed"):
        read_pcd(str(path))


def test_unknown_data_format_is_refused(tmp_path):
    path = tmp_path / "u.pcd"
    path.write_bytes(_header(XYZI, 1, "quantum").encode())
    with pytest.raises(ValueError, match="quantum"):
        read_pcd(str(path))


def test_truncated_binary_payload_is_refused(tmp_path):
    path = tmp_path / "trunc.pcd"
    with open(path, "wb") as fh:
        fh.write(_header(XYZI, 4, "binary").encode())
        fh.write(np.zeros(6, dtype=np.float32).tobytes())  # 1.5 points of 4
    with pytest.raises(ValueError, match="Truncated"):
        read_pcd(str(path))


def test_header_without_data_line_is_refused(tmp_path):
    path = tmp_path / "nodata.pcd"
    path.write_text("VERSION 0.7\nFIELDS x\nSIZE 4\nTYPE F\n")
    with pytest.raises(ValueError, match="Truncated PCD header"):
        read_pcd(str(path))


def test_inconsistent_header_counts_are_refused(tmp_path):
    path = tmp_path / "bad.pcd"
    path.write_text("VERSION 0.7\nFIELDS x y\nSIZE 4\nTYPE F F\nDATA ascii\n")
    with pytest.raises(ValueError, match="Inconsistent"):
        read_pcd(str(path))


def test_unsupported_type_size_pair_is_refused(tmp_path):
    path = tmp_path / "odd.pcd"
    path.write_text("VERSION 0.7\nFIELDS x\nSIZE 3\nTYPE F\nCOUNT 1\nDATA ascii\n1.0\n")
    with pytest.raises(ValueError, match="Unsupported PCD field"):
        read_pcd(str(path))


# ---------------------------------------------------------------- the loader


def test_loader_is_lazy_and_frame_ordered(tmp_path):
    for i, row in enumerate(ROWS):
        write_ascii(tmp_path / f"{i:06d}.pcd", XYZI, [row])
    loader = PCDLoader(str(tmp_path))
    assert len(loader) == 2
    assert loader.shape == (4,)
    assert loader.field_names == ("x", "y", "z", "intensity")
    np.testing.assert_allclose(loader[1], [ROWS[1]])


def test_loader_honours_the_dataset_file_order(tmp_path):
    for i, row in enumerate(ROWS):
        write_ascii(tmp_path / f"{i:06d}.pcd", XYZI, [row])
    loader = PCDLoader(str(tmp_path), files=["000001.pcd", "000000.pcd"])
    np.testing.assert_allclose(loader[0], [ROWS[1]])


def test_declared_fields_keep_the_width_stable_across_differing_headers(tmp_path):
    """The point of `fields`: frame 1 carries an extra field, the channel does not."""
    write_ascii(tmp_path / "000000.pcd", XYZI, [ROWS[0]])
    write_ascii(tmp_path / "000001.pcd", XYZI + [("ring", "U", 4)], [[*ROWS[1], 3.0]])
    loader = PCDLoader(str(tmp_path), fields=["x", "y", "z"])
    assert loader.shape == (3,)
    assert loader[0].shape == loader[1].shape == (1, 3)


def test_frame_missing_a_declared_field_raises_at_access(tmp_path):
    write_ascii(tmp_path / "000000.pcd", XYZI, [ROWS[0]])
    loader = PCDLoader(str(tmp_path), fields=["x", "ring"])
    with pytest.raises(ValueError, match="ring"):
        loader[0]


def test_empty_directory_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="No .pcd frames"):
        PCDLoader(str(tmp_path))


def test_registered_in_both_registries(tmp_path):
    assert str_to_loader["pcd"] is PCDLoader
    path = tmp_path / "d.pcd"
    write_ascii(path, XYZI, ROWS)
    np.testing.assert_allclose(
        DERIVED_LOADERS["pcd"](path), np.array(ROWS, dtype=np.float32)
    )
