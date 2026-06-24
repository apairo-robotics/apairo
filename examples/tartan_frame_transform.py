"""Move TartanDrive lidar points into the robot's base frame.

A lidar scan is expressed in the sensor's own frame. To fuse it with anything
mounted elsewhere on the robot — or to reason about height above the ground — you
first re-express the points in a common frame (here `base_link`). That needs the
static extrinsic: where the lidar sits on the vehicle.

apairo splits this cleanly. It *resolves* the static transform tree — there is
exactly one way to compose fixed edges, so `calibration.get_tf` lives in the core
and hands you the 4x4. *Applying* that matrix to data is the data-dependent part
(points move differently from poses or normals), so it stays a small transform
you register with .transform(). The pipeline reads as: resolve, then apply.

A `/tf_static`-aware extraction populates calibration for you. A bare public
download usually does not, so this also shows the other half: write the extrinsic
you know with register_static_transform() — once in .apairo/calibration.yaml, it
is resolved transparently on every later load.

Calibration lives on every dataset, not just the generic RawDataset:
TartanKittiDataset reads its extrinsics from .apairo/calibration.yaml the same way
it reads channels from .apairo/channels.yaml.
"""

import os
from pathlib import Path

import numpy as np

from apairo import TartanKittiDataset
from apairo.core.config import register_static_transform

SEQ_DIR = Path(os.environ.get("APAIRO_TARTAN_SEQ", "/data/tartan/2024-01-01_forest"))

# Extrinsic from the platform's calibration sheet / URDF: the velodyne is mounted
# 1.2 m above base_link, no rotation. The matrix maps velodyne coords into
# base_link, so the edge is parent=base_link, child=velodyne.
T_base_from_velodyne = np.eye(4)
T_base_from_velodyne[:3, 3] = [0.0, 0.0, 1.2]
register_static_transform(SEQ_DIR, "base_link", "velodyne", T_base_from_velodyne)

ds = TartanKittiDataset(SEQ_DIR, keys=["velodyne_0"])

# Resolve: the core walks the tree (inverting the edge as needed) and returns
# T_base_link_from_velodyne. p_base = T @ p_velodyne.
T = ds.calibration.get_tf("velodyne", "base_link")


def to_base_frame(pts):
    """Apply the rigid transform to the xyz columns, keeping any extra columns."""
    xyz = pts[:, :3] @ T[:3, :3].T + T[:3, 3]
    return np.hstack([xyz, pts[:, 3:]]) if pts.shape[1] > 3 else xyz


# transform() registers in place and returns the same object, so grab the
# in-sensor-frame cloud now, before the transform is in the pipeline.
raw = ds[0].data["velodyne_0"]

ds_base = ds.transform("velodyne_0", to_base_frame)
base = ds_base[0].data["velodyne_0"]

print(f"frames        : {len(ds)}")
print(f"mean z (lidar): {raw[:, 2].mean():+.3f} m")
print(f"mean z (base) : {base[:, 2].mean():+.3f} m")  # raised by the 1.2 m mount
