"""Load a TUM RGB-D sequence in place and associate depth and pose to each colour frame.

The Kinect's colour and depth streams, the motion-capture trajectory and the
accelerometer each run on their own clock. TUM ships ``associate.py`` to pair
every colour image with the depth image closest in time, within 20 ms;
``synchronize(method="nearest", tolerance=0.02)`` is that association for every
channel at once, and it reports how far each match is from the colour frame.

The sequence is read as downloaded: the declaration in
``examples/declarations/tum_rgbd.yaml`` says where each clock lives.
"""

import os
from pathlib import Path

import numpy as np

from apairo import RawDataset

SEQ_DIR = os.environ.get("APAIRO_TUM_SEQ", "/data/tum/rgbd_dataset_freiburg1_xyz")
DECLARATION = Path(__file__).parent / "declarations" / "tum_rgbd.yaml"

ds = RawDataset(SEQ_DIR, declare=DECLARATION)
for key in ds.keys:
    ts = ds.timestamps[key]
    rate = (len(ts) - 1) / (ts[-1] - ts[0])
    print(f"{key:<14} {len(ts):>6} frames at {rate:6.1f} Hz")

# Every colour frame, with the depth image, pose and accelerometer reading
# closest in time. A frame with no match within 20 ms in some channel is
# dropped rather than served with stale data.
frames = ds.synchronize(reference="rgb", method="nearest", tolerance=0.02)
n_rgb = len(ds.timestamps["rgb"])
print(f"\nColour frames      : {n_rgb}")
print(
    f"Associated frames  : {len(frames)}  ({n_rgb - len(frames)} dropped by the tolerance)"
)

for key in ("depth", "groundtruth", "accelerometer"):
    dt = np.abs(frames.time_offsets(key)) * 1000
    print(f"{key:<14} |dt| mean {dt.mean():5.2f} ms, max {dt.max():5.2f} ms")

# `nearest` matches each colour frame on its own, so one depth image may serve
# two colour frames; associate.py pairs one-to-one. frame_indices says which
# event backs every frame, so the reuse is visible rather than silent.
depth_rows = frames.frame_indices["depth"]
print(f"Depth images reused: {len(depth_rows) - len(np.unique(depth_rows))}")

sample = frames[0]
depth_m = sample.data["depth"] / 5000.0  # TUM depth PNGs: 5000 units per metre
tx, ty, tz = sample.data["groundtruth"][:3]
print(f"\nFirst frame at t = {sample.timestamp:.6f}")
print(f"rgb {sample.data['rgb'].shape}, depth {sample.data['depth'].shape}")
print(f"Depth range        : {depth_m[depth_m > 0].min():.2f} to {depth_m.max():.2f} m")
print(f"Camera position    : ({tx:.3f}, {ty:.3f}, {tz:.3f}) m")
