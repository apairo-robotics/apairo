import os
import numpy as np
import matplotlib.pyplot as plt


def load_timestamps(file):
    return np.loadtxt(file)


def get_frequency(timestamps: np.array) -> float:
    """Return the frequency of the timestamps"""
    return 1 / np.mean(timestamps[1:] - timestamps[:-1])


def get_duration(timestamps: np.array) -> float:
    """Return the offset of the timestamps"""
    return timestamps[-1] - timestamps[0]


def get_offset(timestamps: np.array) -> float:
    """Return the offset of the timestamps"""
    return timestamps[0]


def get_end_of_time(timestamps: dict) -> float:
    """Return the end of the time of the timestamps"""
    return max([timestamps[key][-1] for key in timestamps])


def get_reference_timestamps(timestamps: dict) -> str:
    """Return the key of the timestamps with the lowest frequency"""
    freq = {key: get_frequency(value) for key, value in timestamps.items()}
    return min(freq, key=freq.get)


def get_indexes(timestamps: dict, reference=None) -> dict:
    """
    Given the timestamps of each fields, return the indexes for each batch.

    The reference is used to create a new batch for each iteration of his timestamp.

    If there is no reference, the timestamp with the lowest frequency is used.
    This way, the batch will be alaways full with new data.
    We will consider the data that begin after the first timestamp of the reference.

    Args:
        timestamps: dict[str, np.ndarray(float)]
            Field name and timestamps associated
        reference: str
            Field name used as reference for the batch creation

    Returns:
        dict[str, np.ndarray(int)]: Indexes for each field
            Indexes is an array of the latest index for each batch.

    Example:
    >>> timestamps = {
        'a': np.array([0, 3, 6, 9]),
        'b': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        }
    >>> get_indexes(timestamps)
    {'a': array([0, 1, 2, 3], 'b': array([0, 3, 6, 9])}
    """
    if reference is None:
        reference = get_reference_timestamps(timestamps)
    indexes = {key: np.searchsorted(value, timestamps[reference]) for key, value in timestamps.items()}
    return indexes


def show_timestamps(timestamps: dict, index_max=None):
    """Show the timestamps of each fields"""
    if index_max is not None:
        ref = get_reference_timestamps(timestamps)
        time_end = timestamps[ref][index_max]

    for i, (key, value) in enumerate(timestamps.items()):
        if index_max is not None:
            value = value[value < time_end]
        plt.plot(value, i * np.ones_like(value), label=key, marker='o', linestyle='None')
    plt.legend()
    plt.show()


def show_indexed_timestamps(timestamps: dict, indexes: dict, index_max=None):
    """Show the timestamps of each fields
    Args:
        timestamps: dict[str, np.ndarray(float)]
            Field name and timestamps associated
        indexes: dict[str, np.ndarray(int)]
            Field name and last indexes for each batch
        index_max: int
            Maximum index to display from the reference timestamps
    """
    if index_max is not None:
        ref = get_reference_timestamps(timestamps)
        time_end = timestamps[ref][index_max]
    timestamps_indexed = {key: [value[indexes[key][i - 1]: indexes[key][i]]
                                for i in range(1, len(indexes[key]))] for key, value in timestamps.items()}

    colors = ['b', 'g']
    for i, (key, value) in enumerate(timestamps_indexed.items()):
        for j, subvalue in enumerate(value):
            if index_max is not None and len(subvalue) > 0 and subvalue[0] > time_end:
                break

            plt.plot(
                subvalue,
                i * np.ones_like(subvalue), marker='o', linestyle='None', color=colors[j % 2])

    plt.show()


if __name__ == '__main__':
    example_dir_path = "example/tartan2kitti_path"
    timestamps = {}
    for folder_name in os.listdir(example_dir_path):
        path = os.path.join(example_dir_path, folder_name)
        if not os.path.isdir(path):
            continue
        try:
            timestamps[folder_name] = np.loadtxt(os.path.join(path, "timestamps.txt"))
        except FileNotFoundError:
            print(f"File not found: {os.path.join(path, 'timestamps.txt')}")
            print(f"{os.listdir(path)[:5]}")
            continue

    indexes = get_indexes(timestamps)
    show_timestamps(timestamps, index_max=2)
    show_indexed_timestamps(timestamps, indexes, index_max=10)
