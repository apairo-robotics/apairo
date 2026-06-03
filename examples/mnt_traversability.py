"""Compute and persist per-frame traversability labels from MNT trajectory data.

Defines a SequencePreprocessor that marks each frame as traversable (1) or not
(0) based on whether the robot actually traversed it: if a future waypoint
falls within a given radius of the current position, the current frame is
traversable.

Usage:
    python examples/mnt_traversability.py --mission /output/my_dataset/mission_001
"""

import argparse
from pathlib import Path
from typing import Iterator

import numpy as np

from apairo import MNTDataset, SequencePreprocessor
from apairo.core.sample import Sample


class TraversabilityFromTrajectory(SequencePreprocessor):
    """Binary traversability label from the recorded trajectory.

    A frame at position p_i is *traversable* if the robot reached a
    future position within ``radius`` metres -- meaning the robot
    physically drove through that area.

    Output: uint8 array of shape (N,), 1 = traversable.
    """

    output_key = "trav_label"
    output_loader = "npy"
    input_keys = ["position"]
    sources = ["position"]

    def __init__(self, radius: float = 0.5) -> None:
        self.radius = radius

    def process(self, frames: Iterator[Sample]) -> np.ndarray:
        positions = np.stack([s.data["position"] for s in frames])  # (N, 2)
        n = len(positions)
        labels = np.zeros(n, dtype=np.uint8)

        for i in range(n):
            dists = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
            if len(dists) > 0 and dists.min() <= self.radius:
                labels[i] = 1

        frac = labels.mean() * 100
        print(f"Traversable: {labels.sum()}/{n} frames ({frac:.1f} %)")
        return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", required=True, help="Mission directory")
    parser.add_argument("--radius", type=float, default=0.5, help="Traversal radius (m)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    preprocessor = TraversabilityFromTrajectory(radius=args.radius)
    MNTDataset.run_preprocess(preprocessor, args.mission, overwrite=args.overwrite)

    # Verify: reload with the new channel
    ds = MNTDataset(args.mission, keys=["position", "trav_label"])
    sample = ds[0]
    print(f"trav_label[0]: {sample.data['trav_label']}")


if __name__ == "__main__":
    main()
