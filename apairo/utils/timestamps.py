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
