"""Basic usage of MNTDataset -- load and iterate over MNT pipeline output.

The MNT pipeline (AMIAD_MNT_Dataset) produces mission directories in Zarr
format.  MNTDataset wraps them as a standard apairo dataset.

Usage:
    python examples/mnt_basic.py --mission /output/my_dataset/mission_001

    # Or pass a dataset root (all missions):
    python examples/mnt_basic.py --mission /output/my_dataset
"""

import argparse
from pathlib import Path

import numpy as np

from apairo import MNTDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", required=True, help="Mission dir or dataset root")
    args = parser.parse_args()

    path = Path(args.mission)

    # Describe what is available (raw + preprocessed channels)
    MNTDataset.describe(path)

    # Load a few channels
    ds = MNTDataset(path, keys=["position", "yaw", "points"])
    print(f"Total frames : {len(ds)}")
    print(f"Missions     : {ds.mission_ids}")

    # Access the first frame
    sample = ds[0]
    print(f"position[0]  : {sample.data['position']}")
    print(f"yaw[0]       : {sample.data['yaw']:.4f} rad")
    pts = sample.data["points"]
    print(f"points[0]    : shape {pts.shape}, dtype {pts.dtype}")

    # Iterate over a single mission as a SequenceView
    for seq in ds.sequences():
        print(f"\nMission '{seq.name}' -- {len(seq)} frames")
        for i, sample in enumerate(seq):
            pos = sample.data["position"]
            _ = pos  # do something
            if i >= 3:
                break


if __name__ == "__main__":
    main()
