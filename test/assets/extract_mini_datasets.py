"""Extract committed mini-datasets from full local copies (maintainers only).

Builds the two real-data fixtures used by the smoke tests:

- ``mini_rellis``  -- synchronous  (Rellis-3D layout: bin + label + poses + lst splits)
- ``mini_tartan``  -- asynchronous (TartanDrive v2 KITTI layout: multi-rate channels)

Point clouds are subsampled to keep the repository small (~300 KB total).
Source paths point to the lab storage; re-run only if the fixtures need to be
regenerated.

Usage::

    python test/assets/extract_mini_datasets.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

VAULT = Path("/mnt/vault-fellowship")
RELLIS_SRC = VAULT / "rellis" / "Rellis-3D"
TARTAN_SRC = VAULT / "tartan" / "2023-11-14-15-02-21_figure_8"

ASSETS = Path(__file__).parent
RELLIS_DST = ASSETS / "mini_rellis"
TARTAN_DST = ASSETS / "mini_tartan" / "figure_8"

N_POINTS_RELLIS = 1024
N_POINTS_TARTAN = 512
RELLIS_SEQS = ["00000", "00001"]
RELLIS_FRAMES = 5
TARTAN_FIRST_FRAME = 100
TARTAN_FRAMES = 8


def _subsample(n_total: int, n_keep: int) -> np.ndarray:
    """Evenly spaced indices so the mini cloud spans the full scan."""
    return np.linspace(0, n_total - 1, n_keep).astype(np.intp)


def extract_rellis() -> None:
    if RELLIS_DST.exists():
        shutil.rmtree(RELLIS_DST)

    for seq in RELLIS_SEQS:
        src = RELLIS_SRC / seq
        dst = RELLIS_DST / "Rellis-3D" / seq
        (dst / "os1_cloud_node_kitti_bin").mkdir(parents=True)
        (dst / "os1_cloud_node_semantickitti_label_id").mkdir()

        for i in range(RELLIS_FRAMES):
            stem = f"{i:06d}"
            pts = np.fromfile(
                src / "os1_cloud_node_kitti_bin" / f"{stem}.bin", dtype=np.float32
            ).reshape(-1, 4)
            labels = np.fromfile(
                src / "os1_cloud_node_semantickitti_label_id" / f"{stem}.label",
                dtype=np.int32,
            )
            assert len(pts) == len(labels), f"{seq}/{stem}: point/label mismatch"
            keep = _subsample(len(pts), N_POINTS_RELLIS)
            pts[keep].tofile(dst / "os1_cloud_node_kitti_bin" / f"{stem}.bin")
            labels[keep].tofile(
                dst / "os1_cloud_node_semantickitti_label_id" / f"{stem}.label"
            )

        with open(src / "poses.txt") as fh:
            poses = [next(fh) for _ in range(RELLIS_FRAMES)]
        (dst / "poses.txt").write_text("".join(poses))
        shutil.copy(src / "calib.txt", dst / "calib.txt")

    def lst_line(seq: str, i: int) -> str:
        return (
            f"{seq}/os1_cloud_node_kitti_bin/{i:06d}.bin "
            f"{seq}/os1_cloud_node_semantickitti_label_id/{i:06d}.label\n"
        )

    (RELLIS_DST / "pt_train.lst").write_text(
        "".join(lst_line("00000", i) for i in range(RELLIS_FRAMES))
    )
    (RELLIS_DST / "pt_val.lst").write_text(
        "".join(lst_line("00001", i) for i in range(3))
    )
    (RELLIS_DST / "pt_test.lst").write_text(
        "".join(lst_line("00001", i) for i in range(3, RELLIS_FRAMES))
    )


def extract_tartan() -> None:
    if TARTAN_DST.parent.exists():
        shutil.rmtree(TARTAN_DST.parent)

    vel_ts = np.loadtxt(TARTAN_SRC / "velodyne_0" / "timestamps.txt")
    k0, k1 = TARTAN_FIRST_FRAME, TARTAN_FIRST_FRAME + TARTAN_FRAMES
    t0, t1 = vel_ts[k0], vel_ts[k1 - 1]

    # velodyne_0: per-frame npy pairs (points + intensity), renumbered from 0
    vel_dst = TARTAN_DST / "velodyne_0"
    vel_dst.mkdir(parents=True)
    for new_i, src_i in enumerate(range(k0, k1)):
        pts = np.load(TARTAN_SRC / "velodyne_0" / f"{src_i:06d}.npy")
        intensity = np.load(TARTAN_SRC / "velodyne_0" / f"{src_i:06d}_intensity.npy")
        keep = _subsample(len(pts), N_POINTS_TARTAN)
        np.save(vel_dst / f"{new_i:06d}.npy", pts[keep])
        np.save(vel_dst / f"{new_i:06d}_intensity.npy", intensity[keep])
    np.savetxt(vel_dst / "timestamps.txt", vel_ts[k0:k1], fmt="%.18e")

    # cmd / multisense_imu: single stacked npy, rows cut to the same window
    # (with margin before t0 so method="previous" has an event for frame 0)
    for channel, stacked in [("cmd", "twist.npy"), ("multisense_imu", "imu.npy")]:
        ts = np.loadtxt(TARTAN_SRC / channel / "timestamps.txt")
        rows = np.load(TARTAN_SRC / channel / stacked)
        mask = (ts >= t0 - 0.2) & (ts <= t1 + 0.01)
        dst = TARTAN_DST / channel
        dst.mkdir(parents=True)
        np.save(dst / stacked, rows[mask])
        np.savetxt(dst / "timestamps.txt", ts[mask], fmt="%.18e")


if __name__ == "__main__":
    extract_rellis()
    extract_tartan()
    total = sum(f.stat().st_size for f in ASSETS.rglob("*") if f.is_file())
    print(f"mini datasets written under {ASSETS} ({total / 1024:.0f} KB total)")
