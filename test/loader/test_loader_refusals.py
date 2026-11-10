"""Loaders refuse, or say, what would otherwise give wrong data quietly."""

from __future__ import annotations

import io
import tarfile

import numpy as np
import pytest

from apairo import RawDataset
from apairo.loader import TarImageLoader, load_timestamps
from apairo.loader.pcd_loader import read_pcd
from apairo.utils.naming import integer_frame_index

HEADER = (
    "VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
    "WIDTH {n}\nHEIGHT 1\nPOINTS {n}\nDATA ascii\n"
)


def test_a_clock_file_holds_one_time_per_line(tmp_path):
    path = tmp_path / "timestamps.txt"
    np.savetxt(path, np.c_[np.arange(3), np.arange(3)])
    with pytest.raises(ValueError, match="2 columns.*one time per line"):
        load_timestamps(path)
    np.savetxt(path, np.arange(3) * 0.1)
    assert load_timestamps(path).shape == (3,)
    np.savetxt(path, [0.5])
    assert load_timestamps(path).shape == (1,)


def test_a_channel_named_like_a_format_still_needs_a_clock(tmp_path):
    for name in ("lidar", "npy"):
        d = tmp_path / name
        d.mkdir()
        for i in range(3):
            np.save(d / f"{i:06d}.npy", np.full(2, float(i)))
    np.savetxt(tmp_path / "lidar" / "timestamps.txt", np.arange(3) * 0.1)
    with pytest.raises(ValueError, match="No timestamps.txt for 'npy'"):
        RawDataset(tmp_path)


@pytest.mark.parametrize("rows", [2, 8])
def test_an_ascii_pcd_must_hold_the_points_it_declares(tmp_path, rows):
    path = tmp_path / "cloud.pcd"
    path.write_text(HEADER.format(n=5) + "1 2 3\n" * rows)
    with pytest.raises(ValueError, match=f"{rows} point.*declares 5"):
        read_pcd(str(path))


def test_an_ascii_pcd_with_its_points_loads(tmp_path):
    path = tmp_path / "cloud.pcd"
    path.write_text(HEADER.format(n=3) + "1 2 3\n" * 3)
    assert read_pcd(str(path)).shape == (3, 3)
    path.write_text(HEADER.format(n=0))
    assert read_pcd(str(path)).shape == (0, 3)


def test_a_binary_pcd_without_a_point_count_is_refused(tmp_path):
    path = tmp_path / "cloud.pcd"
    header = b"FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nDATA binary\n"
    path.write_bytes(header + np.arange(9, dtype=np.float32).tobytes())
    with pytest.raises(ValueError, match="neither POINTS nor WIDTH"):
        read_pcd(str(path))


def test_tar_images_keep_32_bit_and_float_values(tmp_path):
    image = pytest.importorskip("PIL.Image")
    depth = np.arange(16, dtype=np.int32).reshape(4, 4) * 4000
    arrays = {"0.tif": depth, "1.tif": depth.astype(np.float32) / 7}
    with tarfile.open(tmp_path / "images.tar", "w") as tf:
        for name, array in arrays.items():
            buf = io.BytesIO()
            image.fromarray(array).save(buf, format="TIFF")
            info = tarfile.TarInfo(name)
            info.size = buf.tell()
            buf.seek(0)
            tf.addfile(info, buf)
    loader = TarImageLoader(tmp_path / "images.tar", 2, integer_frame_index)
    np.testing.assert_array_equal(loader[0][..., 0], depth)  # 60000 stays 60000
    np.testing.assert_allclose(loader[1][..., 0], arrays["1.tif"])
