import numpy as np


def load_timestamps(file):
    # atleast_1d: a single-frame channel's timestamps.txt is one line, which
    # np.loadtxt returns as a 0-d scalar -- callers index/iterate it as a 1-d array.
    return np.atleast_1d(np.loadtxt(file))


def get_frequency(timestamps: np.ndarray) -> float:
    """Return the mean frequency of the timestamps in Hz."""
    return 1 / np.mean(timestamps[1:] - timestamps[:-1])


def get_end_of_time(timestamps: dict) -> float:
    """Return the latest timestamp across all channels."""
    return max([timestamps[key][-1] for key in timestamps])


def get_reference_timestamps(timestamps: dict) -> str:
    """Return the key of the channel with the lowest frequency."""
    freq = {key: get_frequency(value) for key, value in timestamps.items()}
    return min(freq, key=freq.get)


def merge_timeline(timestamps: dict, keys: list) -> tuple:
    """Merge per-channel timestamps into one timestamp-ordered event timeline.

    Returns ``(key_idxs, frame_idxs)``: for timeline position ``i``, the
    event is frame ``frame_idxs[i]`` of channel ``keys[key_idxs[i]]``.
    Stable sort -- events keep channel-declaration order on equal timestamps.
    """
    lengths = [len(np.atleast_1d(timestamps[k])) for k in keys]
    all_ts = np.concatenate([np.atleast_1d(timestamps[k]).astype(float) for k in keys])
    key_idxs = np.repeat(np.arange(len(keys), dtype=np.intp), lengths)
    frame_idxs = np.concatenate([np.arange(n, dtype=np.intp) for n in lengths])
    order = np.argsort(all_ts, kind="stable")
    return key_idxs[order], frame_idxs[order]


def clock_from_distance(
    timestamps: np.ndarray,
    positions: np.ndarray,
    step: float,
) -> np.ndarray:
    """Build a reference clock that ticks every *step* metres travelled.

    Given an odometry stream (timestamps + positions), returns the timestamp
    of the first sample at or past each multiple of *step* along the
    cumulative path length.  Pass the result as the ``reference`` of
    :meth:`~apairo.core.abstract_dataset.AbstractDataset.synchronize` to
    resample a dataset spatially instead of temporally::

        odom_ts  = ds.timestamps["odom"]
        odom_xy  = np.stack([rows[:, 0], rows[:, 1]], axis=1)
        clock    = clock_from_distance(odom_ts, odom_xy, step=0.5)
        ds_sync  = ds.synchronize(reference=clock)   # one frame every 0.5 m

    A stationary robot produces no ticks -- static periods are skipped by
    construction.

    Args:
        timestamps: ``(N,)`` ascending timestamps of the odometry samples.
        positions: ``(N, D)`` positions (any dimensionality, typically xy
            or xyz) aligned with *timestamps*.
        step: Distance between two ticks, in the unit of *positions*.

    Returns:
        ``(M,)`` ascending timestamps, deduplicated.
    """
    timestamps = np.asarray(timestamps, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if len(timestamps) != len(positions):
        raise ValueError(
            f"timestamps ({len(timestamps)}) and positions ({len(positions)}) "
            f"must have the same length."
        )
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}.")

    cumdist = np.zeros(len(positions))
    cumdist[1:] = np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    targets = np.arange(0.0, cumdist[-1] + step, step)
    idx = np.searchsorted(cumdist, targets, side="left")
    idx = np.unique(idx[idx < len(timestamps)])
    return timestamps[idx]
