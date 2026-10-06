"""CSVLoader: one delimited text table per channel, one row per frame, the clock
optionally in a column. Fixtures reproduce the two headers the loader was built
for -- EuRoC's ``#timestamp [ns],...`` comment header and TUM's whitespace
``groundtruth.txt`` -- plus a plain named header row."""

import numpy as np
import pytest

from apairo.core.keys import parse_column_key
from apairo.core.utils.exceptions import FileExtensionError
from apairo.loader import CSVLoader

EUROC = (
    "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
    "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n"
    "1403636579758555392,-0.099134701513277898,0.14730578886832138,"
    "0.02722713633111154,8.1476917083333333,-0.37592158333333331,-2.4026292499999999\n"
    "1403636579763555584,-0.099134701513277898,0.14032447186034408,"
    "0.029321531433504364,8.033280791666666,-0.40861041666666664,-2.4026292499999999\n"
)

TUM = (
    "# ground truth trajectory\n"
    "# file: 'rgbd_dataset_freiburg1_xyz.bag'\n"
    "# timestamp tx ty tz qx qy qz qw\n"
    "1305031098.6659 1.3563 0.6305 1.6380 0.6132 0.5962 -0.3311 -0.3986\n"
    "1305031098.6758 1.3543 0.6306 1.6360 0.6129 0.5966 -0.3316 -0.3980\n"
)


def _table(tmp_path, text, name="data.csv"):
    d = tmp_path / "chan"
    d.mkdir()
    (d / name).write_text(text)
    return d


def test_euroc_comment_header_names_columns_and_units_are_dropped(tmp_path):
    loader = CSVLoader(_table(tmp_path, EUROC), key_column="timestamp")
    assert loader.columns == [
        "w_RS_S_x",
        "w_RS_S_y",
        "w_RS_S_z",
        "a_RS_S_x",
        "a_RS_S_y",
        "a_RS_S_z",
    ]
    assert len(loader) == 2
    assert loader.shape == (6,)
    assert loader.key_tokens == ["1403636579758555392", "1403636579763555584"]
    np.testing.assert_allclose(loader[0][3], 8.1476917083333333)


def test_tum_whitespace_table_named_by_last_comment(tmp_path):
    d = _table(tmp_path, TUM, name="groundtruth.txt")
    loader = CSVLoader(d, file="groundtruth.txt", key_column=0)
    assert loader.columns == ["tx", "ty", "tz", "qx", "qy", "qz", "qw"]
    np.testing.assert_allclose(
        loader[1], [1.3543, 0.6306, 1.6360, 0.6129, 0.5966, -0.3316, -0.3980]
    )


def test_plain_header_row_and_fields_projection(tmp_path):
    d = _table(tmp_path, "t,x,y,z\n0.0,1,2,3\n0.1,4,5,6\n")
    loader = CSVLoader(d, key_column="t", fields=["z", "x"])
    assert loader.columns == ["z", "x"]
    np.testing.assert_allclose(loader[1], [6.0, 4.0])


def test_headerless_table_keeps_every_column_but_the_key(tmp_path):
    d = _table(tmp_path, "0.0\t1\t2\n0.1\t3\t4\n")
    loader = CSVLoader(d, key_column=0)
    assert loader.columns is None
    np.testing.assert_allclose(loader[0], [1.0, 2.0])
    no_key = CSVLoader(d)
    assert no_key.shape == (3,)
    assert no_key.key_tokens is None


def test_rows_are_copies(tmp_path):
    loader = CSVLoader(_table(tmp_path, "1,2\n3,4\n"))
    row = loader[0]
    row[:] = 0
    np.testing.assert_allclose(loader[0], [1.0, 2.0])


def test_comment_that_does_not_fit_the_width_is_not_a_header(tmp_path):
    loader = CSVLoader(_table(tmp_path, "# recorded on the rig\n1,2,3\n"))
    assert loader.columns is None


@pytest.mark.parametrize(
    "text, kwargs, match",
    [
        ("1,2\n3\n", {}, "expected 2"),
        ("a,b\n", {}, "no data rows"),
        ("1,2\n", {"key_column": 5}, "out of range"),
        ("t,x\n1,2\n", {"key_column": "time"}, "no column named 'time'"),
        ("1,2\n", {"fields": ["x"]}, "no header"),
        ("t,x\n1,oops\n", {}, "non-numeric"),
    ],
)
def test_malformed_tables_fail_naming_the_file(tmp_path, text, kwargs, match):
    with pytest.raises(ValueError, match=match):
        CSVLoader(_table(tmp_path, text), **kwargs)


def test_file_selection(tmp_path):
    d = tmp_path / "chan"
    d.mkdir()
    with pytest.raises(FileExtensionError, match="No .csv"):
        CSVLoader(d)
    (d / "a.csv").write_text("1\n")
    (d / "b.csv").write_text("2\n")
    with pytest.raises(FileExtensionError, match="Several .csv"):
        CSVLoader(d)
    np.testing.assert_allclose(CSVLoader(d, file="b.csv")[0], [2.0])
    with pytest.raises(FileExtensionError, match="No such table"):
        CSVLoader(d, file="c.csv")


# ───────────────────────────── column key parsing ─────────────────────────────


def test_nanosecond_integer_key_is_scaled_exactly_once():
    keys = parse_column_key(["1403636579758555392"], {"column": 0, "units": ["ns"]})
    assert keys[0] == pytest.approx(1403636579.758555392, abs=1e-6)


def test_seconds_key_without_units():
    keys = parse_column_key(["1305031098.6659", "1305031098.6758"], {"column": 0})
    np.testing.assert_allclose(keys, [1305031098.6659, 1305031098.6758])


@pytest.mark.parametrize(
    "spec, match",
    [
        ({"column": 0, "units": ["ns", "s"]}, "one 'units'/'scale' entry"),
        ({"column": 0, "units": ["h"]}, "unknown key unit"),
        ({"column": 0, "units": ["s"], "scale": [1.0]}, "both 'units' and 'scale'"),
    ],
)
def test_column_key_spec_errors(spec, match):
    with pytest.raises(ValueError, match=match):
        parse_column_key(["1"], spec)


def test_non_numeric_key_cell():
    with pytest.raises(ValueError, match="non-numeric key cell"):
        parse_column_key(["2026-09-24"], {"column": 0})


# ─────────────── scan_table: status's size and span, no cell parsed ───────────

_SCANNED = [
    (EUROC, {"key_column": 0}),
    (TUM, {"key_column": "timestamp", "fields": ["tx", "qw"]}),
    ("t,x,y\n1,2,3\n\n# a note in the middle\n2,3,4\n   \n  # indented\n3,4,5", {}),
    ("t,x\r\n1,2\r\n\r\n2,3\r\n", {"key_column": "t"}),
    ("1\t2\t3\n4\t5\t6\n", {"key_column": 2}),
    ("# no header\n1 2\n3 4\n5 6\n", {"key_column": 1}),
]


@pytest.mark.parametrize("text, kwargs", _SCANNED)
def test_scan_agrees_with_the_loader(tmp_path, text, kwargs):
    from apairo.loader.csv_loader import scan_table

    d = _table(tmp_path, text)
    loader = CSVLoader(d, **kwargs)
    scan = scan_table(d, **kwargs)
    assert scan.rows == len(loader)
    assert scan.width == loader.shape[0]
    assert scan.columns == loader.columns
    if loader.key_tokens is not None:
        assert (scan.first_key, scan.last_key) == (
            loader.key_tokens[0],
            loader.key_tokens[-1],
        )


@pytest.mark.parametrize(
    "text, kwargs, match",
    [
        ("1,2\n3,4\n5\n", {}, "expected 2"),
        ("a,b\n", {}, "no data rows"),
        ("t,x\n1,2\n", {"key_column": "time"}, "no column named 'time'"),
        ("1,2\n", {"fields": ["x"]}, "no header"),
    ],
)
def test_scan_refuses_what_the_loader_refuses(tmp_path, text, kwargs, match):
    from apairo.loader.csv_loader import scan_table

    with pytest.raises(ValueError, match=match):
        scan_table(_table(tmp_path, text), **kwargs)
