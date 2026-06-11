import numpy as np


def load_timestamps(file):
    return np.loadtxt(file)


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
    cumdist[1:] = np.cumsum(
        np.linalg.norm(np.diff(positions, axis=0), axis=1)
    )
    targets = np.arange(0.0, cumdist[-1] + step, step)
    idx = np.searchsorted(cumdist, targets, side="left")
    idx = np.unique(idx[idx < len(timestamps)])
    return timestamps[idx]
