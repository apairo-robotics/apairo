"""Rebuild the samples of the standard datasets that are publicly downloadable.

Each sample is a few frames read straight out of the dataset's official
archive, with HTTP range requests: the archive's directory is read, then only
the members needed. Clouds and labels are strided to ``N_POINTS`` points,
the same indices for both, so every label stays on its point.

- ``mini_goose``: GOOSE 3D (https://goose-dataset.de, CC BY-SA 4.0). Three
  scans of two validation scenes and of one training scene, clouds and labels
  real, file names and tree as in the archives.
- ``mini_semantic_kitti``: SemanticKITTI (http://www.semantic-kitti.org,
  CC BY-NC-SA 3.0). Three label files of sequences 00 and 08 are real. The
  clouds and ``times.txt`` come from the KITTI odometry archives, which need a
  registration, so they are synthetic here, with the real point counts.
- ``mini_kuka_ft_imu``: the KUKA LBR Med F/T and IMU logs (Skrede, Zenodo
  10.5281/zenodo.11096791, CC BY 4.0). The ``1-baseline`` tables whole, and
  the first second of the two other test runs; the static calibration tables,
  which carry no clock, are left out.

Usage::

    python test/assets/fetch_public_samples.py
"""

from __future__ import annotations

import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).parent
N_POINTS = 1024

GOOSE = {
    "https://goose-dataset.de/storage/goose_3d_val.zip": [
        ("val", "2023-05-15_neubiberg_rain", 3),
        ("val", "2022-08-30_siegertsbrunn_feldwege", 3),
    ],
    "https://goose-dataset.de/storage/goose_3d_train.zip": [
        ("train", "2023-01-20_campus_snow", 3),
    ],
}
SEMANTIC_KITTI_LABELS = "http://www.semantic-kitti.org/assets/data_odometry_labels.zip"
SEMANTIC_KITTI = [("00", 3), ("08", 3)]
KUKA = "https://zenodo.org/records/11096791/files/{name}?download=1"
KUKA_RUNS = {"1-baseline": None, "2-vibrations": 1.0, "3-vibrations-contact": 1.0}


class _RangeFile(io.RawIOBase):
    """A remote file read with HTTP range requests, seekable as zipfile needs."""

    def __init__(self, url: str) -> None:
        head = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(head, timeout=60) as r:
            self.url, self.size = r.geturl(), int(r.headers["Content-Length"])
        self.pos = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, buffer) -> int:
        if not len(buffer) or self.pos >= self.size:
            return 0
        end = min(self.pos + len(buffer), self.size) - 1
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.pos}-{end}"}
        )
        with urllib.request.urlopen(request, timeout=120) as r:
            data = r.read()
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def _archive(url: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BufferedReader(_RangeFile(url), buffer_size=1 << 20))


def _keep(n_total: int) -> np.ndarray:
    """Evenly spaced indices, so the strided cloud spans the whole scan."""
    return np.linspace(0, n_total - 1, N_POINTS).astype(np.intp)


def fetch_goose(dst: Path = ASSETS / "mini_goose") -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    for url, scenes in GOOSE.items():
        z = _archive(url)
        names = z.namelist()
        for split, scene, n in scenes:
            clouds = sorted(x for x in names if x.startswith(f"lidar/{split}/{scene}/"))
            labels = sorted(
                x for x in names if x.startswith(f"labels/{split}/{scene}/")
            )
            for cloud, label in zip(clouds[:n], labels[:n], strict=True):
                pts = np.frombuffer(z.read(cloud), np.float32).reshape(-1, 4)
                lab = np.frombuffer(z.read(label), np.uint32)
                assert len(pts) == len(lab), (cloud, label)
                keep = _keep(len(pts))
                for member, data in ((cloud, pts[keep]), (label, lab[keep])):
                    out = dst / member
                    out.parent.mkdir(parents=True, exist_ok=True)
                    data.tofile(out)
    return dst


def fetch_semantic_kitti(dst: Path = ASSETS / "mini_semantic_kitti") -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    z = _archive(SEMANTIC_KITTI_LABELS)
    rng = np.random.default_rng(0)
    for seq, n in SEMANTIC_KITTI:
        root = dst / "sequences" / seq
        (root / "labels").mkdir(parents=True)
        (root / "velodyne").mkdir()
        for i in range(n):
            lab = np.frombuffer(
                z.read(f"dataset/sequences/{seq}/labels/{i:06d}.label"), np.uint32
            )
            lab[_keep(len(lab))].tofile(root / "labels" / f"{i:06d}.label")
            # Synthetic: the KITTI velodyne archive needs a registration.
            cloud = np.column_stack(
                [rng.uniform(-50, 50, (N_POINTS, 3)), rng.uniform(0, 1, N_POINTS)]
            ).astype(np.float32)
            cloud.tofile(root / "velodyne" / f"{i:06d}.bin")
        # Synthetic too, in KITTI's format: seconds from the start, ~10 Hz.
        np.savetxt(root / "times.txt", np.arange(n) * 0.1, fmt="%e")
    return dst


def fetch_kuka(dst: Path = ASSETS / "mini_kuka_ft_imu") -> Path:
    """Each run's three tables, whole or cut after *seconds* of rows: a cut
    keeps the sampling rates, which a stride would not."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir()
    for run, seconds in KUKA_RUNS.items():
        for table in ("accel", "wrench", "orientations"):
            name = f"{run}_{table}.csv"
            with urllib.request.urlopen(KUKA.format(name=name), timeout=120) as r:
                data = r.read()
            if seconds is None:  # whole, byte for byte
                (dst / name).write_bytes(data)
                continue
            lines = data.decode().splitlines()
            t0 = int(lines[1].split(",")[0])
            kept = [
                row for row in lines[1:] if int(row.split(",")[0]) - t0 <= seconds * 1e6
            ]
            (dst / name).write_text("\n".join([lines[0], *kept]) + "\n")
    return dst


if __name__ == "__main__":
    for path in (fetch_goose(), fetch_semantic_kitti(), fetch_kuka()):
        print("wrote", path.relative_to(ASSETS))
