"""Pose-path resampling onto external clocks.

Counterpart of ``ds.synchronize(reference=...)`` for *array-level* work:
resample a recorded 2D pose path (positions + yaw) at arbitrary targets
along a monotonic axis — time, arc-length, or anything ascending.

Unlike ``synchronize`` (which drops non-bracketed ticks), targets outside
the recorded range are **clamped** to the boundary pose, so the output
always has ``len(targets)`` entries — the contract trajectory horizons
need.  Yaw is interpolated circularly (shortest path) and returned wrapped
to ``[-pi, pi]``.
"""

from __future__ import annotations

import numpy as np


def cumulative_distance(positions: np.ndarray) -> np.ndarray:
    """Cumulative arc-length along a 2D polyline (same length as input)."""
    n = len(positions)
    cd = np.zeros(n, dtype=np.float64)
    if n > 1:
        diffs = np.diff(np.asarray(positions, dtype=np.float64), axis=0)
        cd[1:] = np.cumsum(np.sqrt((diffs * diffs).sum(axis=1)))
    return cd


def resample_pose_path(
    axis: np.ndarray,
    positions: np.ndarray,
    yaws: np.ndarray,
    targets: np.ndarray,
    timestamps: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Linearly resample a 2D pose path at *targets* along *axis*.

    Args:
        axis: (N,) ascending parameterisation of the path (timestamps for
            temporal resampling, :func:`cumulative_distance` for spatial).
        positions: (N, 2) path positions.
        yaws: (N,) path headings (radians).
        targets: (M,) axis values to sample at; out-of-range targets are
            clamped to the boundary pose.
        timestamps: Optional (N,) series interpolated alongside (used by
            spatial resampling to recover the time of each arc-length mark).

    Returns:
        ``(positions (M, 2), yaws (M,), timestamps (M,) or None)`` float64.
    """
    axis = np.asarray(axis, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    if len(axis) == 0:
        ts = np.zeros(len(targets)) if timestamps is not None else None
        return (
            np.zeros((len(targets), 2), dtype=np.float64),
            np.zeros(len(targets), dtype=np.float64),
            ts,
        )

    out_pos = np.stack(
        [
            np.interp(targets, axis, positions[:, 0]),
            np.interp(targets, axis, positions[:, 1]),
        ],
        axis=1,
    )

    # Circular yaw: unwrap -> linear interp -> wrap back to [-pi, pi].
    # Unwrapping takes the shortest path between consecutive samples, which
    # is the per-segment circular interpolation convention.
    unwrapped = np.unwrap(np.asarray(yaws, dtype=np.float64))
    out_yaw = np.interp(targets, axis, unwrapped)
    out_yaw = np.arctan2(np.sin(out_yaw), np.cos(out_yaw))

    out_ts = None
    if timestamps is not None:
        out_ts = np.interp(targets, axis, np.asarray(timestamps, dtype=np.float64))

    return out_pos, out_yaw, out_ts
