"""Load a EuRoC MAV sequence in place and put the IMU and ground truth on the camera clock.

The stereo cameras run at 20 Hz, the IMU and the ground-truth state at 200 Hz,
each stamped in nanoseconds: in the image filenames for the cameras, in the
first column of a ``data.csv`` for the others. ``synchronize()`` aligns
everything onto the left camera and says how far each match is from the frame.

The sequence is read as downloaded: a declaration that ships with apairo says
where each clock lives. There is one per hall, because the Machine Hall
sequences carry a laser tracker (``leica0``) and the Vicon Room ones a motion
capture system (``vicon0``).
"""

import os
from pathlib import Path

import numpy as np

from apairo import RawDataset

SEQ_DIR = Path(os.environ.get("APAIRO_EUROC_SEQ", "/data/euroc/V1_01_easy/mav0"))
HALL = "vicon_room" if (SEQ_DIR / "vicon0").is_dir() else "machine_hall"
DECLARATION = f"euroc_{HALL}"  # a declaration that ships with apairo, by name

ds = RawDataset(SEQ_DIR, declare=DECLARATION)
for key in ds.keys:
    ts = ds.timestamps[key]
    rate = (len(ts) - 1) / (ts[-1] - ts[0])
    print(f"{key:<12} {len(ts):>6} frames at {rate:6.1f} Hz")

# Every left-camera frame, with the right image, the IMU reading and the
# ground-truth state closest in time. The ground truth starts after the cameras
# and stops before them: frames outside it have no match within 5 ms and are
# dropped rather than paired with a state from another moment.
vio = RawDataset(
    SEQ_DIR, keys=["cam0", "cam1", "imu0", "groundtruth"], declare=DECLARATION
)
frames = vio.synchronize(reference="cam0", method="nearest", tolerance=0.005)
n_cam = len(vio.timestamps["cam0"])
print(f"\nCamera frames       : {n_cam}")
print(
    f"Synchronized frames : {len(frames)}  ({n_cam - len(frames)} outside the ground truth)"
)

for key in ("cam1", "imu0", "groundtruth"):
    dt = np.abs(frames.time_offsets(key)) * 1000
    print(f"{key:<12} |dt| mean {dt.mean():5.3f} ms, max {dt.max():5.3f} ms")

sample = frames[0]
gyro, accel = sample.data["imu0"][:3], sample.data["imu0"][3:]
position = sample.data["groundtruth"][:3]
print(f"\nFirst frame at t = {sample.timestamp:.6f}")
print(f"cam0 {sample.data['cam0'].shape}, cam1 {sample.data['cam1'].shape}")
print(f"Angular rate        : {np.round(gyro, 4)} rad/s")
print(f"Acceleration        : {np.round(accel, 3)} m/s^2")
print(f"Position            : {np.round(position, 3)} m")
