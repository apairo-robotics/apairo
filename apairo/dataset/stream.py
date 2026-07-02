from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from apairo.core import AbstractDataset, Sample


class StreamDataset(AbstractDataset):
    """In-memory asynchronous dataset built from timestamped event streams.

    The bridge between live or freshly-decoded data (ROS messages, queue
    items, arrays in RAM) and the apairo API: give it one ``(timestamps,
    items)`` pair per channel and it behaves exactly like a file-backed
    asynchronous dataset -- merged timeline, single-event samples, and above
    all :meth:`~apairo.core.abstract_dataset.AbstractDataset.synchronize`::

        ds = StreamDataset({
            "image": (img_ts, img_msgs),      # any indexable items
            "lidar": (lidar_ts, lidar_msgs),
            "odom":  (odom_ts,  odom_msgs),
        })

        ds[0]                                  # one event, timestamp-ordered
        frames = ds.synchronize(reference=clock, method="previous")

    Items are stored as given -- they can be numpy arrays, ROS messages, or
    any Python objects; apairo never copies or converts them.

    Args:
        streams: ``{channel: (timestamps, items)}``.  Timestamps must be
            ascending 1-D arrays; ``len(items)`` must match.

    Raises:
        ValueError: On empty streams, length mismatch, or non-ascending
            timestamps.
    """

    def __init__(
        self,
        streams: Dict[str, Tuple[np.ndarray, Sequence]],
    ) -> None:
        if not streams:
            raise ValueError("StreamDataset requires at least one stream.")

        self.loaders: Dict[str, Sequence] = {}
        self.timestamps: Dict[str, np.ndarray] = {}
        for key, (ts, items) in streams.items():
            ts = np.asarray(ts, dtype=np.float64)
            if ts.ndim != 1 or len(ts) == 0:
                raise ValueError(
                    f"Stream {key!r}: timestamps must be a non-empty 1-D "
                    f"array, got shape {ts.shape}."
                )
            if len(ts) != len(items):
                raise ValueError(
                    f"Stream {key!r}: {len(ts)} timestamps for "
                    f"{len(items)} items."
                )
            if np.any(np.diff(ts) < 0):
                raise ValueError(f"Stream {key!r}: timestamps must be ascending.")
            self.timestamps[key] = ts
            self.loaders[key] = items

        self._set_keys(list(streams))

        from apairo.utils.timestamps import merge_timeline
        self._tl_key_idxs, self._tl_frame_idxs = merge_timeline(
            self.timestamps, self._keys
        )

    def __len__(self) -> int:
        return len(self._tl_key_idxs)

    def _load(self, idx: int) -> Sample:
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        key = self._keys[self._tl_key_idxs[idx]]
        frame = int(self._tl_frame_idxs[idx])
        return Sample(
            data={key: self.loaders[key][frame]},
            timestamp=float(self.timestamps[key][frame]),
        )

    def __repr__(self) -> str:
        sizes = {k: len(v) for k, v in self.loaders.items()}
        return f"StreamDataset(events={len(self)}, streams={sizes})"
