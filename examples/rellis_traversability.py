"""Build the `trav_gt` channel for RELLIS-3D — the prerequisite for the rest.

A fresh RELLIS download ships `lidar` and semantic `labels`, but no
traversability ground truth. The other RELLIS examples here (training pipeline,
k-fold, cached prior, cross-dataset concat) all consume a `trav_gt` channel that
is *derived* from those labels.

This is the script that derives it. The point of doing it through apairo rather
than a one-off numpy loop: the derivation becomes an explicit, declarative step.
The FramePreprocessor states its contract — read `labels`, write `trav_gt`, one
value per point — and run_preprocess() handles the file naming, saving and
.apairo registration. Downstream, `trav_gt` then loads like any raw channel: the
consuming examples never see the derivation, they just ask for the key.
"""

import os
from pathlib import Path

import numpy as np

from apairo import Rellis3DDataset, FramePreprocessor
from apairo.core.sample import Sample

# "Traversable" is a labeling decision you own, not something apairo defines.
# Here: the RELLIS-3D ground classes a wheeled robot can drive over. The IDs come
# straight from the RELLIS-3D ontology — edit the set to change what counts.
TRAVERSABLE = {
    1:  "dirt",
    3:  "grass",
    10: "asphalt",
    23: "concrete",
    31: "puddle",
    33: "mud",
}


class TraversabilityFromLabels(FramePreprocessor):
    """Per-point traversability (uint8: 1 = traversable) from semantic labels."""

    output_key      = "trav_gt"
    output_loader   = "npys"     # one file per frame, aligned with the lidar points
    input_keys      = ["labels"]
    timestamps_from = "lidar"
    sources         = ["labels"]

    def __call__(self, sample: Sample) -> np.ndarray:
        return np.isin(sample.data["labels"], list(TRAVERSABLE)).astype(np.uint8)


root = Path(os.environ.get("APAIRO_RELLIS_ROOT", "/data/RELLIS"))

# The one-time build step: compute trav_gt and persist it under .apairo. overwrite
# rebuilds in place, so re-running after editing TRAVERSABLE just refreshes it.
Rellis3DDataset(root, keys=["lidar", "labels"]).run_preprocess(
    TraversabilityFromLabels(), overwrite=True
)

# From here on trav_gt is a normal channel — apairo reloads it transparently, with
# no preprocessor in sight. It is point-aligned with the lidar, so the traversable
# point cloud is one mask away: slice the xyz columns by the channel.
ds = Rellis3DDataset(root, keys=["lidar", "trav_gt"])
pts, trav = ds[0].data["lidar"], ds[0].data["trav_gt"]
traversable_xyz = pts[trav == 1][:, :3]

print(f"frames           : {len(ds)}")
print(f"points / frame   : {len(pts)}")
print(f"traversable frac : {trav.mean():.1%}")
print(f"traversable cloud: {traversable_xyz.shape}")
