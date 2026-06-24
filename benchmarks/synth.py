"""Synthetic apairo datasets of a parametrised size.

Mirrors the realistic rosbag -> kitti -> RawDataset path: an async, multi-rate
sequence (a lidar, a faster pose stream). The same dataset exercises per-frame
access, synchronize, and the lazy views. Generated trees are cached under
``benchmarks/.cache`` and reused across runs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

CACHE = Path(__file__).parent / ".cache"


def make_async_sequence(n_ref_frames: int, points: int, ref_rate: float = 20.0) -> Path:
    """A lidar at ``ref_rate`` Hz, shape (points, 4); a pose stream at 5x, shape (7,).

    Returns the sequence root, cached by (n_ref_frames, points)."""
    root = CACHE / f"async_n{n_ref_frames}_p{points}"
    if not (root / "lidar" / "timestamps.txt").exists():
        rng = np.random.default_rng(0)
        duration = n_ref_frames / ref_rate
        channels = {"lidar": (ref_rate, (points, 4)), "pose": (ref_rate * 5, (7,))}
        for name, (rate, shape) in channels.items():
            d = root / name
            d.mkdir(parents=True, exist_ok=True)
            n = int(round(duration * rate))
            for i in range(n):
                np.save(d / f"{i:06d}.npy", rng.standard_normal(shape).astype(np.float32))
            np.savetxt(d / "timestamps.txt", np.arange(n) / rate)

    if not (root / ".apairo" / "channels.yaml").exists():
        from apairo.dataset.raw import RawDataset

        RawDataset.init(root)
    return root
