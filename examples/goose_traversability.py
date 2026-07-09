"""Compute and persist per-point traversability labels from a GOOSE3D dataset.

Defines a FramePreprocessor that maps semantic labels to a binary traversability
mask (0 / 1) using a YAML config. The runner handles file naming, saving, and
.apairo registration automatically via run_preprocess.

Usage:
    python examples/goose_traversability.py \
        --root /data/goose/GOOSE_3D \
        --config examples/goose_traversable_labels.yaml
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml

from apairo import Goose3DDataset, FramePreprocessor
from apairo.core.sample import Sample


TRAVERSABLE_LABELS = {
    23,  # Asphalt
    24,  # Gravel
    31,  # Soil
    50,  # Low grass
    51,  # High grass
}


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
        self._traversable: set[int] = set(
            cfg.get("traversable_map") or TRAVERSABLE_LABELS
        )

    def __call__(self, sample: Sample) -> np.ndarray:
        labels: np.ndarray = sample.data["labels"]  # (N,)
        mask = np.zeros(len(labels), dtype=bool)
        for lbl in self._traversable:
            mask |= labels == lbl
        return mask.astype(np.uint8)


def main():
    logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", required=True, help="GOOSE split root (e.g. GOOSE_3D/train)"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute even if output already exists",
    )
    args = parser.parse_args()

    root_dir = Path(args.root).resolve()
    preprocessor = TraversabilityPreprocessor(args.config)

    logging.info("Config          : %s", Path(args.config).resolve())
    logging.info("Traversable IDs : %s", sorted(preprocessor._traversable))
    logging.info("Dataset root    : %s", root_dir)
    logging.info(
        "Output key      : %s  (format: %s)",
        preprocessor.output_key,
        preprocessor.output_loader,
    )
    logging.info("")

    Goose3DDataset.run_preprocess(preprocessor, root_dir, overwrite=args.overwrite)

    logging.info("")
    logging.info(
        "Channel '%s' registered in %s/.apairo", preprocessor.output_key, root_dir
    )


if __name__ == "__main__":
    main()
