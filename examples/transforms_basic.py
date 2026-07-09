"""At-access transforms: per-channel, sample-level, and filter patterns.

Demonstrates the three core usage patterns of dataset.transform():

  1. Per-channel  -- normalise a single modality
  2. Sample-level -- consistent mask across aligned channels (filter)
  3. Publish + keep=False -- intermediate channel used by downstream steps

No data is written to disk; transforms run in memory at __getitem__ time.

Usage:
    python examples/transforms_basic.py --root /data/goose/GOOSE_3D
"""

import argparse
import numpy as np
from apairo import Goose3DDataset, Compose


# ---------------------------------------------------------------------------
# 1. Per-channel transform
# ---------------------------------------------------------------------------


def example_per_channel(root: str) -> None:
    """Intensity normalisation applied to the lidar channel only."""

    ds = Goose3DDataset(root, keys=["lidar"])
    ds.transform("lidar", lambda pts: pts / (pts.max() + 1e-6))

    sample = ds[0]
    print("[per-channel] lidar max after norm:", sample.data["lidar"].max())


# ---------------------------------------------------------------------------
# 2. Sample-level transform (filter)
# ---------------------------------------------------------------------------


def example_filter(root: str) -> None:
    """Range filter applied consistently to lidar AND its aligned labels.

    Both channels share the same index structure -- filtering one without
    the other would misalign them.  The sample-level form gives access to
    all channels at once so the mask is computed once and applied to both.
    """

    ds = Goose3DDataset(root, keys=["lidar", "labels"])

    def range_filter(sample):
        pts = sample.data["lidar"]
        dist = np.linalg.norm(pts[:, :3], axis=1)
        mask = dist < 50.0
        sample.data["lidar"] = pts[mask]
        sample.data["labels"] = sample.data["labels"][mask]
        return sample

    ds.transform(range_filter)

    sample = ds[0]
    n_pts = sample.data["lidar"].shape[0]
    print(f"[filter] {n_pts} points within 50 m")
    assert sample.data["lidar"].shape[0] == sample.data["labels"].shape[0]


# ---------------------------------------------------------------------------
# 3. Published channel + keep=False (temporary intermediate)
# ---------------------------------------------------------------------------


def example_publish(root: str) -> None:
    """Filtered lidar published as a shared source for two independent branches.

    'lidar_f' is available to all subsequent steps but removed from the
    final sample (keep=False) so the caller only sees 'lidar_norm' and
    'lidar_vox'.
    """

    def voxel_downsample(pts: np.ndarray, voxel: float = 0.2) -> np.ndarray:
        keys = (pts[:, :3] / voxel).astype(int)
        _, idx = np.unique(keys, axis=0, return_index=True)
        return pts[idx]

    def normalize(pts: np.ndarray) -> np.ndarray:
        return pts / (pts.max() + 1e-6)

    ds = Goose3DDataset(root, keys=["lidar"])

    # Step 1 -- range filter, publish result as 'lidar_f', drop it at the end
    ds.transform(
        "lidar",
        lambda pts: pts[np.linalg.norm(pts[:, :3], axis=1) < 50.0],
        output="lidar_f",
        keep=False,
    )

    # Steps 2a and 2b -- two independent branches from the same filtered source
    ds.transform("lidar_f", normalize, output="lidar_norm")
    ds.transform("lidar_f", voxel_downsample, output="lidar_vox")

    sample = ds[0]
    assert "lidar_f" not in sample.data, "'lidar_f' should have been dropped"
    print(f"[publish] lidar_norm: {sample.data['lidar_norm'].shape}")
    print(f"[publish] lidar_vox : {sample.data['lidar_vox'].shape}")


# ---------------------------------------------------------------------------
# 4. Compose -- named pipeline for reuse
# ---------------------------------------------------------------------------


def example_compose(root: str) -> None:
    """Bundle several per-channel ops into a named Compose object."""

    ds = Goose3DDataset(root, keys=["lidar"])
    ds.transform(
        "lidar",
        Compose(
            [
                lambda pts: pts[np.linalg.norm(pts[:, :3], axis=1) < 50.0],
                lambda pts: pts / (pts.max() + 1e-6),
            ]
        ),
    )

    sample = ds[0]
    print(
        f"[compose] {sample.data['lidar'].shape[0]} points, max={sample.data['lidar'].max():.4f}"
    )


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="GOOSE_3D root directory")
    args = parser.parse_args()

    example_per_channel(args.root)
    example_filter(args.root)
    example_publish(args.root)
    example_compose(args.root)


if __name__ == "__main__":
    main()
