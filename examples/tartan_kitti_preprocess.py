"""Register and run a custom preprocessing pipeline on a TartanDrive sequence."""

import os

import numpy as np
from apairo.preprocess import FramePreprocessor, SequencePreprocessor
from apairo.dataset import TartanKittiDataset
from apairo.core.sample import Sample

SEQ_DIR = os.environ.get("APAIRO_TARTAN_SEQ", "/data/tartan/2024-01-01_forest")


# --- Frame-by-frame preprocessor (one output file per scan) ----------------


class TravLabel(FramePreprocessor):
    """Assign a traversability score to each LiDAR point."""

    output_key = "trav_label"
    output_loader = "npys"
    input_keys = ["velodyne_0"]
    timestamps_from = "velodyne_0"  # no own timestamps.txt needed
    sources = ["velodyne_0"]

    def process(self, sample: Sample) -> np.ndarray:
        pts = sample.data["velodyne_0"]  # (N, 4)
        # Replace with your actual model / heuristic:
        return np.zeros(len(pts), dtype=np.float32)


# --- Sequence-level preprocessor (one output file for the whole sequence) --


class GICPPoses(SequencePreprocessor):
    """Estimate per-frame poses via ICP over the full sequence."""

    output_key = "gicp_poses"
    output_loader = "npy"
    input_keys = ["velodyne_0"]
    sources = ["velodyne_0"]

    def process(self, frames):
        poses = []
        for sample in frames:
            # Replace with your actual ICP call:
            poses.append(np.eye(4, dtype=np.float32))
        return np.stack(poses)  # (N, 4, 4)


# --- Run -------------------------------------------------------------------

TartanKittiDataset.run_preprocess(TravLabel(), SEQ_DIR)
TartanKittiDataset.run_preprocess(GICPPoses(), SEQ_DIR)

# Load the sequence including preprocessed channels
ds = TartanKittiDataset(SEQ_DIR, keys=["velodyne_0", "trav_label", "gicp_poses"])
print("Keys   :", ds.keys)
print("Length :", len(ds))
