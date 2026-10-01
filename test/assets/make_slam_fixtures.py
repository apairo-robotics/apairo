"""Build the synthetic TUM RGB-D and EuRoC MAV layout fixtures.

Unlike ``mini_rellis`` and ``mini_tartan``, these hold **no real data**: every
value is generated here. What they reproduce is each dataset's *layout* -- the
directory tree, the file naming, the header lines and the timestamp units -- so
the declarations in ``examples/declarations/`` and the guide examples are
exercised in CI against the shape of the real thing.

- ``mini_tum/rgbd_dataset_freiburg1_xyz``  -- images named ``<sec>.<usec>.png``,
  whitespace tables with a ``# timestamp ...`` header comment at the sequence root
- ``mini_euroc/V1_01_easy/mav0``           -- images named ``<ns>.png`` under
  ``cam*/data``, one ``data.csv`` per sensor, a Vicon tracker
- ``mini_euroc/MH_05_difficult/mav0``      -- the same with a Leica tracker

The headers were copied from the real sequences of the same names; the
declarations were checked against those sequences by hand (see the guides).

Usage::

    python test/assets/make_slam_fixtures.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image

ASSETS = Path(__file__).parent
TUM_DST = ASSETS / "mini_tum" / "rgbd_dataset_freiburg1_xyz"
EUROC_DST = ASSETS / "mini_euroc"

IMG_H, IMG_W = 6, 8

# ── TUM RGB-D ────────────────────────────────────────────────────────────────
TUM_BAG = "# file: 'rgbd_dataset_freiburg1_xyz.bag'"
TUM_T0 = 1305031102.175304  # first colour frame of the real sequence
TUM_FRAMES = 6
# Depth lags colour by a few ms; the third depth frame is 30 ms off, so its
# colour frame has no depth within the 20 ms the guide tolerates.
TUM_DEPTH_LAG = [0.004, -0.003, 0.030, 0.006, -0.002, 0.005]


def _stamp(t: float) -> str:
    return f"{t:.6f}"


def make_tum(dst: Path = TUM_DST) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "rgb").mkdir(parents=True)
    (dst / "depth").mkdir()

    rgb_t = [TUM_T0 + i / 30 for i in range(TUM_FRAMES)]
    depth_t = [t + lag for t, lag in zip(rgb_t, TUM_DEPTH_LAG, strict=True)]

    rgb_index = ["# color images", TUM_BAG, "# timestamp filename"]
    for i, t in enumerate(rgb_t):
        name = f"rgb/{_stamp(t)}.png"
        Image.fromarray(np.full((IMG_H, IMG_W, 3), 40 * i, np.uint8)).save(dst / name)
        rgb_index.append(f"{_stamp(t)} {name}")
    (dst / "rgb.txt").write_text("\n".join(rgb_index) + "\n")

    depth_index = ["# depth maps", TUM_BAG, "# timestamp filename"]
    for i, t in enumerate(depth_t):
        name = f"depth/{_stamp(t)}.png"
        # 16-bit depth, 5000 units per metre: 1.0 m, 1.2 m, ...
        depth = np.full((IMG_H, IMG_W), 5000 + 1000 * i, np.uint16)
        Image.fromarray(depth).save(dst / name)
        depth_index.append(f"{_stamp(t)} {name}")
    (dst / "depth.txt").write_text("\n".join(depth_index) + "\n")

    # Motion capture at 100 Hz, starting before the first image as in the real file.
    gt = ["# ground truth trajectory", TUM_BAG, "# timestamp tx ty tz qx qy qz qw"]
    for k in range(30):
        t = TUM_T0 - 0.02 + k / 100
        gt.append(
            f"{t:.4f} {1.3 + 0.01 * k:.4f} 0.6300 1.6400 0.6132 0.5962 -0.3311 -0.3986"
        )
    (dst / "groundtruth.txt").write_text("\n".join(gt) + "\n")

    # Accelerometer at 500 Hz.
    acc = ["# accelerometer data", TUM_BAG, "# timestamp ax ay az"]
    for k in range(120):
        t = TUM_T0 - 0.02 + k / 500
        acc.append(f"{t:.6f} {-0.48 + 0.001 * k:.6f} 8.010560 -4.166928")
    (dst / "accelerometer.txt").write_text("\n".join(acc) + "\n")
    return dst


# ── EuRoC MAV (ASL format) ───────────────────────────────────────────────────
IMU_HEADER = (
    "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
    "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]"
)
STATE_HEADER = (
    "#timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], "
    "q_RS_y [], q_RS_z [], v_RS_R_x [m s^-1], v_RS_R_y [m s^-1], v_RS_R_z [m s^-1], "
    "b_w_RS_S_x [rad s^-1], b_w_RS_S_y [rad s^-1], b_w_RS_S_z [rad s^-1], "
    "b_a_RS_S_x [m s^-2], b_a_RS_S_y [m s^-2], b_a_RS_S_z [m s^-2]"
)
VICON_HEADER = (
    "#timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],"
    "q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z []"
)
LEICA_HEADER = "#timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m]"

EUROC_FRAMES = 5
CAM_PERIOD_NS = 50_000_000  # 20 Hz
IMU_PERIOD_NS = 5_000_000  # 200 Hz


def _table(path: Path, header: str, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([header, *rows]) + "\n")


def make_euroc(name: str, t0: int, tracker: str, dst_root: Path = EUROC_DST) -> Path:
    """One sequence. *tracker* is ``"vicon0"`` (Vicon Room) or ``"leica0"``
    (Machine Hall); *t0* is the first camera stamp, in nanoseconds."""
    mav0 = dst_root / name / "mav0"
    if mav0.parent.exists():
        shutil.rmtree(mav0.parent)

    cam_t = [t0 + i * CAM_PERIOD_NS for i in range(EUROC_FRAMES)]
    for c, cam in enumerate(("cam0", "cam1")):
        (mav0 / cam / "data").mkdir(parents=True)
        for i, t in enumerate(cam_t):
            image = np.full((IMG_H, IMG_W), 50 * i + 5 * c, np.uint8)
            Image.fromarray(image).save(mav0 / cam / "data" / f"{t}.png")
        _table(
            mav0 / cam / "data.csv",
            "#timestamp [ns],filename",
            [f"{t},{t}.png" for t in cam_t],
        )

    # IMU at 200 Hz over the whole camera span: row k carries k as angular rate.
    n_imu = (EUROC_FRAMES - 1) * (CAM_PERIOD_NS // IMU_PERIOD_NS) + 1
    _table(
        mav0 / "imu0" / "data.csv",
        IMU_HEADER,
        [f"{t0 + k * IMU_PERIOD_NS},{k},0.0,0.0,9.81,0.0,0.0" for k in range(n_imu)],
    )

    # The ground-truth state starts after the cameras, as in the real sequences:
    # the first two camera frames have no state within the guide's tolerance.
    first = 2 * (CAM_PERIOD_NS // IMU_PERIOD_NS)
    state = []
    for k in range(first, n_imu):
        p = f"{0.01 * k:.6f},2.142470,0.947262"
        state.append(
            f"{t0 + k * IMU_PERIOD_NS},{p},0.060514,-0.828459,-0.058956,-0.553641,"
            "0.009474,-0.014009,-0.002145,-0.002229,0.020700,0.076350,"
            "-0.012492,0.547666,0.069073"
        )
    _table(mav0 / "state_groundtruth_estimate0" / "data.csv", STATE_HEADER, state)

    if tracker == "vicon0":  # motion capture at 100 Hz: position + quaternion
        rows = [
            f"{t0 + k * 10_000_000},{0.02 * k:.6f},2.1766,1.0620,0.9932,-0.0093,0.0227,0.1137"
            for k in range(21)
        ]
        _table(mav0 / "vicon0" / "data.csv", VICON_HEADER, rows)
    elif tracker == "leica0":  # laser tracker at about 15 Hz: position only
        rows = [
            f"{t0 + k * 68_000_000},{4.39 + 0.01 * k:.6f},-1.7321,0.6953"
            for k in range(4)
        ]
        _table(mav0 / "leica0" / "data.csv", LEICA_HEADER, rows)
    else:
        raise ValueError(f"unknown tracker {tracker!r}")
    return mav0


if __name__ == "__main__":
    tum = make_tum()
    vicon = make_euroc("V1_01_easy", 1403715273262142976, "vicon0")
    hall = make_euroc("MH_05_difficult", 1403638518077829376, "leica0")
    for path in (tum, vicon, hall):
        print("wrote", path.relative_to(ASSETS))
