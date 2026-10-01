"""``status`` reports each channel's rate, span and shape from what the layout
already says, without loading the data: a clock declared as a ``key`` (parsed
from the filenames, or read from a sidecar) and the first frame of an ``img`` /
``bin`` / ``pcd`` channel. It used to print ``-`` and ``?`` for both."""

import json

import numpy as np
import pytest
from PIL import Image

from apairo.cli import main
from apairo.core.config import write_config
from test.loader.test_pcd_loader import XYZI, write_binary


def _status(path, capsys):
    try:
        main(["status", str(path), "--json"])
    except SystemExit as exc:
        assert exc.code in (0, None)
    return json.loads(capsys.readouterr().out)["channels"]


def _config(root, **channels):
    write_config(
        root,
        {
            "version": 1,
            "channels": {k: {"kind": "raw", **v} for k, v in channels.items()},
        },
    )


def _frames(directory, stems, shape=(4, 3)):
    directory.mkdir(parents=True)
    for stem in stems:
        np.save(directory / f"{stem}.npy", np.zeros(shape, np.float32))


def test_a_filename_key_gives_the_rate_and_the_span(tmp_path, capsys):
    _frames(tmp_path / "camera", [f"frame{i}-{1000 + i * 100}" for i in range(11)])
    _config(
        tmp_path,
        camera={"loader": "npys", "key": {"name": r"frame\d+-(\d+)", "units": ["ms"]}},
    )
    cam = _status(tmp_path, capsys)["camera"]
    assert cam["frames"] == 11
    assert cam["rate_hz"] == pytest.approx(10.0)
    assert cam["span"] == pytest.approx([1.0, 2.0])


def test_the_clock_is_sorted_whatever_the_listing_order(tmp_path, capsys):
    # Lexicographic order would put 1000 before 200: the span must not care.
    _frames(tmp_path / "camera", ["200", "1000", "600"])
    _config(
        tmp_path, camera={"loader": "npys", "key": {"name": r"(\d+)", "units": ["ms"]}}
    )
    assert _status(tmp_path, capsys)["camera"]["span"] == pytest.approx([0.2, 1.0])


def test_a_sidecar_key_gives_the_clock(tmp_path, capsys):
    _frames(tmp_path / "lidar", [f"{i:06d}" for i in range(5)])
    np.savetxt(tmp_path / "lidar" / "stamps.txt", np.arange(5) * 0.5)
    _config(tmp_path, lidar={"loader": "npys", "key": {"file": "stamps.txt"}})
    lidar = _status(tmp_path, capsys)["lidar"]
    assert lidar["rate_hz"] == pytest.approx(2.0)
    assert lidar["span"] == pytest.approx([0.0, 2.0])


def test_a_declared_key_takes_precedence_over_timestamps_txt(tmp_path, capsys):
    """Loading reads the key first; status must describe the same clock."""
    _frames(tmp_path / "camera", ["100", "200", "300"])
    np.savetxt(tmp_path / "camera" / "timestamps.txt", [50.0, 60.0, 70.0])
    _config(
        tmp_path, camera={"loader": "npys", "key": {"name": r"(\d+)", "units": ["ms"]}}
    )
    assert _status(tmp_path, capsys)["camera"]["span"] == pytest.approx([0.1, 0.3])


def test_a_key_that_does_not_parse_leaves_the_rate_blank(tmp_path, capsys):
    _frames(tmp_path / "camera", ["a", "b"])
    _config(tmp_path, camera={"loader": "npys", "key": {"name": r"(\d+)"}})
    cam = _status(tmp_path, capsys)["camera"]
    assert cam["rate_hz"] is None
    assert cam["span"] is None


@pytest.mark.parametrize(
    "array, shape, dtype",
    [
        (np.zeros((6, 8, 3), np.uint8), [6, 8, 3], "uint8"),
        (np.zeros((6, 8), np.uint8), [6, 8], "uint8"),
        (np.zeros((6, 8), np.uint16), [6, 8], "uint16"),
    ],
)
def test_an_image_channel_reports_its_first_frame(
    tmp_path, capsys, array, shape, dtype
):
    (tmp_path / "rgb").mkdir()
    for t in ("1305031102.175304", "1305031102.211214"):
        Image.fromarray(array).save(tmp_path / "rgb" / f"{t}.png")
    _config(tmp_path, rgb={"loader": "img", "key": {"name": r"(\d+)\.(\d+)"}})
    rgb = _status(tmp_path, capsys)["rgb"]
    assert rgb["shape"] == shape
    assert rgb["dtype"] == dtype
    assert rgb["rate_hz"] == pytest.approx(1 / 0.03591, rel=1e-3)


def test_a_bin_channel_reports_its_point_count(tmp_path, capsys):
    (tmp_path / "lidar").mkdir()
    for i in range(2):
        np.zeros((7 + i, 4), np.float32).tofile(tmp_path / "lidar" / f"{i:06d}.bin")
    np.savetxt(tmp_path / "lidar" / "timestamps.txt", [0.0, 0.1])
    _config(tmp_path, lidar={"loader": "bin"})
    lidar = _status(tmp_path, capsys)["lidar"]
    assert lidar["shape"] == [7, 4]
    assert lidar["dtype"] == "float32"


def test_a_pcd_channel_reports_its_declared_fields(tmp_path, capsys):
    (tmp_path / "cloud").mkdir()
    write_binary(tmp_path / "cloud" / "000000.pcd", XYZI, [[1, 2, 3, 4], [5, 6, 7, 8]])
    np.savetxt(tmp_path / "cloud" / "timestamps.txt", [0.0])
    _config(tmp_path, cloud={"loader": "pcd", "fields": ["x", "y", "z"]})
    cloud = _status(tmp_path, capsys)["cloud"]
    assert cloud["shape"] == [2, 3]
    assert cloud["dtype"] == "float32"
