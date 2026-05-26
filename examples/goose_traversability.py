"""Compute and persist per-point traversability labels from a GOOSE3D sequence.

Defines a FramePreprocessor that maps semantic labels to a binary traversability
mask (0 / 1) using a YAML config. The runner handles file naming, saving, and
.apairo registration automatically via run_preprocess.

Usage:
    python examples/goose_traversability.py \
        --seq /data/goose/seq_a \
        --config examples/goose_traversable_labels.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from apairo import Goose3DDataset, FramePreprocessor
from apairo.core.sample import Sample


class TraversabilityPreprocessor(FramePreprocessor):
    """Map GOOSE semantic labels to a binary traversability mask.

    Reads which label IDs are traversable from a YAML file with the key
    ``traversable_map``.  Produces a uint8 array of shape (N,): 1 = traversable.
    """

    output_key = "trav_label"
    output_loader = "npys"
    input_keys = ["labels"]
    timestamps_from = "labels"
    sources = ["labels"]

    def __init__(self, config_path: str | Path) -> None:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self._traversable: set[int] = set(cfg["traversable_map"])

    def process(self, sample: Sample) -> np.ndarray:
        labels: torch.Tensor = sample.data["labels"]  # (N,)
        mask = torch.zeros(len(labels), dtype=torch.bool)
        for lbl in self._traversable:
            mask |= labels == lbl
        return mask.numpy().astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", required=True, help="GOOSE sequence directory")
    parser.add_argument("--config", default="examples/goose_traversable_labels.yaml")
    args = parser.parse_args()

    preprocessor = TraversabilityPreprocessor(args.config)
    print(f"Traversable labels : {sorted(preprocessor._traversable)}")

    Goose3DDataset.run_preprocess(preprocessor, args.seq)


if __name__ == "__main__":
    main()
