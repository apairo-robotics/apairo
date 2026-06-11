"""MNT mission layout — the on-disk format, in one object.

``MNT_LAYOUT`` is the single source of truth for where each channel lives
and how it is encoded (dtype, chunks, compression, member naming).  The
dataset (read path), the extraction pipeline (write path) and the tooling
(slicing, describe) all consume this object.

The table covers the legacy grouped layout (``trajectory.zarr/...``,
``traj_time.zarr/...``).  It is frozen: any new channel follows the flat
convention ``<mission>/<key>.zarr`` through the layout's default spec and
never adds an entry here.
"""

from __future__ import annotations

import numpy as np

from apairo.core.layout import ChannelSpec, DatasetLayout
from apairo.utils.naming import integer_frame_index

_F32, _F64 = np.float32, np.float64


# Chunking policies (frame axis first).
def _frames64(shape: tuple) -> tuple:
    return (min(64, shape[0]), *shape[1:])


def _flat1024(shape: tuple) -> tuple:
    return (min(1024, shape[0]), *shape[1:])


def _per_frame(shape: tuple) -> tuple:
    return (1, *shape[1:])


def _points_chunks(shape: tuple) -> tuple:
    return (min(64, shape[0]), min(16384, shape[1]), shape[2])


def _zarr(path: tuple, dtype, chunks) -> ChannelSpec:
    return ChannelSpec(path=path, dtype=dtype, chunks=chunks)


MNT_LAYOUT = DatasetLayout(
    channels={
        # ── images / lidar ──────────────────────────────────────────────────
        "image": ChannelSpec(
            path=("images.tar",),
            store="tar_jpeg",
            name_to_index=integer_frame_index,
            member_name=lambda i: f"{i:06d}.jpg",
            write_options={"quality": 95},
        ),
        "points": _zarr(("points.zarr",), _F32, _points_chunks),
        # ── current odometry ────────────────────────────────────────────────
        "position":  _zarr(("trajectory.zarr", "positions.zarr"), _F32, _flat1024),
        "yaw":       _zarr(("trajectory.zarr", "yaws.zarr"), _F32, _flat1024),
        "timestamp": _zarr(("trajectory.zarr", "timestamps.zarr"), _F64, _flat1024),
        "timestamp_delta": _zarr(
            ("trajectory.zarr", "timestamps_delta.zarr"), _F64, _flat1024
        ),
        # ── past odometry window ────────────────────────────────────────────
        "position_past":  _zarr(("trajectory.zarr", "positions_past.zarr"), _F32, _per_frame),
        "yaw_past":       _zarr(("trajectory.zarr", "yaws_past.zarr"), _F32, _per_frame),
        "timestamp_past": _zarr(("trajectory.zarr", "timestamps_past.zarr"), _F64, _per_frame),
        "timestamp_delta_past": _zarr(
            ("trajectory.zarr", "timestamps_delta_past.zarr"), _F64, _per_frame
        ),
        # ── future trajectory (temporal sampling) ───────────────────────────
        "waypoints_time":     _zarr(("traj_time.zarr", "positions.zarr"), _F32, _per_frame),
        "yaw_waypoints_time": _zarr(("traj_time.zarr", "yaws.zarr"), _F32, _per_frame),
        "timestamp_waypoints_time": _zarr(
            ("traj_time.zarr", "timestamps.zarr"), _F64, _per_frame
        ),
        "timestamp_delta_waypoints_time": _zarr(
            ("traj_time.zarr", "timestamps_delta.zarr"), _F64, _per_frame
        ),
        "waypoints_time_past":     _zarr(("traj_time.zarr", "positions_past.zarr"), _F32, _per_frame),
        "yaw_waypoints_time_past": _zarr(("traj_time.zarr", "yaws_past.zarr"), _F32, _per_frame),
        "timestamp_waypoints_time_past": _zarr(
            ("traj_time.zarr", "timestamps_past.zarr"), _F64, _per_frame
        ),
        "timestamp_delta_waypoints_time_past": _zarr(
            ("traj_time.zarr", "timestamps_delta_past.zarr"), _F64, _per_frame
        ),
        # ── future trajectory (spatial sampling) ────────────────────────────
        "waypoints_dist":     _zarr(("traj_dist.zarr", "positions.zarr"), _F32, _per_frame),
        "yaw_waypoints_dist": _zarr(("traj_dist.zarr", "yaws.zarr"), _F32, _per_frame),
        "timestamp_waypoints_dist": _zarr(
            ("traj_dist.zarr", "timestamps.zarr"), _F64, _per_frame
        ),
        "timestamp_delta_waypoints_dist": _zarr(
            ("traj_dist.zarr", "timestamps_delta.zarr"), _F64, _per_frame
        ),
        "waypoints_dist_past":     _zarr(("traj_dist.zarr", "positions_past.zarr"), _F32, _per_frame),
        "yaw_waypoints_dist_past": _zarr(("traj_dist.zarr", "yaws_past.zarr"), _F32, _per_frame),
        "timestamp_waypoints_dist_past": _zarr(
            ("traj_dist.zarr", "timestamps_past.zarr"), _F64, _per_frame
        ),
        "timestamp_delta_waypoints_dist_past": _zarr(
            ("traj_dist.zarr", "timestamps_delta_past.zarr"), _F64, _per_frame
        ),
    },
    compression=("zstd", 5),
    # Flat-convention channels (width curves, future additions): float32
    # blocks chunked by 64 frames, like the historical writer defaults.
    default=ChannelSpec(path=(), dtype=_F32, chunks=_frames64),
)
