"""Frames named by plain integers are read in numeric order, padded or not:
by name, 10 would come before 2 and take its timestamp."""

from __future__ import annotations

import numpy as np
import pytest

from apairo import RawDataset
from apairo.core.naming import frame_order

N = 12


def _write(d, i, ext):
    if ext == "npy":
        np.save(d / f"{i}.npy", np.full(2, float(i)))
    elif ext == "bin":
        np.full((1, 4), float(i), dtype=np.float32).tofile(d / f"{i}.bin")
    else:
        (d / f"{i}.pcd").write_text(
            "FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nWIDTH 1\nHEIGHT 1\nPOINTS 1\n"
            f"DATA ascii\n{i} {i} {i}\n"
        )


@pytest.mark.parametrize("ext", ["npy", "bin", "pcd"])
def test_unpadded_frames_keep_their_own_timestamp(tmp_path, ext):
    d = tmp_path / "lidar"
    d.mkdir()
    for i in range(N):
        _write(d, i, ext)
    np.savetxt(d / "timestamps.txt", np.arange(N) * 0.1)
    ds = RawDataset(tmp_path)
    read = [int(np.asarray(ds.loaders["lidar"][i]).flat[0]) for i in range(N)]
    assert read == list(range(N))
    assert list(ds.frame_stems) == [str(i) for i in range(N)]


def test_suffixed_variants_follow_the_same_order(tmp_path):
    d = tmp_path / "lidar"
    d.mkdir()
    for i in range(N):
        np.save(d / f"{i}.npy", np.full(2, float(i)))
        np.save(d / f"{i}_intensity.npy", np.full(1, float(i)))
    np.savetxt(d / "timestamps.txt", np.arange(N) * 0.1)
    ds = RawDataset(tmp_path)
    variant = next(k for k in ds.keys if k.endswith("intensity"))
    assert [int(ds.loaders[variant][i][0]) for i in range(N)] == list(range(N))


def test_frame_order_rules():
    assert frame_order(["10.npy", "2.npy", "1.npy"]) == ["1.npy", "2.npy", "10.npy"]
    assert frame_order(["000010.npy", "000002.npy"]) == ["000002.npy", "000010.npy"]
    # Not all integers: by name, as before.
    assert frame_order(["b10.npy", "b2.npy"]) == ["b10.npy", "b2.npy"]
    assert frame_order(["10.npy", "x.npy", "2.npy"]) == ["10.npy", "2.npy", "x.npy"]
    assert frame_order([]) == []
