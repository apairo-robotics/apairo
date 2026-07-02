"""A competent hand-rolled loader -- the honest baseline apairo must match.

This is what a careful engineer writes without apairo: a sorted glob per channel
and a vectorised latest-match align. Not a strawman; the point is to show apairo
adds no I/O cost on top of this, while giving lazy views and far less code.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class NaiveLoader:
    """Sorted per-frame files in a channel directory, loaded on access."""

    def __init__(self, channel_dir: str | Path) -> None:
        d = Path(channel_dir)
        self.files = sorted(f for f in d.glob("*.npy"))
        self.ts = np.loadtxt(d / "timestamps.txt")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> np.ndarray:
        return np.load(self.files[idx])


def latest_match(ref_ts: np.ndarray, other_ts: np.ndarray) -> np.ndarray:
    """Index into *other* of the last event with t <= each reference tick.

    The same zero-order-hold matching apairo's ``method="previous"`` does."""
    return np.searchsorted(other_ts, ref_ts, side="right") - 1
