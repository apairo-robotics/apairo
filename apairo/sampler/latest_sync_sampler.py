from typing import Dict
import numpy as np

from src.utils.types import Timestamp
from src.utils.timestamps import get_reference_timestamps
from ..core import AbstractSampler


class LatestSyncSampler(AbstractSampler):
    r""" Implement :class:`AbstractSampler` using a update check for each data stream.

    Samples the dataset by batching data when all data streams have been updated.
    It will replace the data in the batch with the latest data available for each data stream
    until all data streams have been updated.

    Be careful, this sampler can give sampler with different shapes and even can miss some data
    if the data taken is at the end of the dataset.
    """

    def __init__(self, timestamp: Timestamp, sample_size: int = 1, shuffle: bool = False):
        super().__init__(timestamp, sample_size=sample_size)
        self.shuffle = shuffle
        self._updated = np.zeros(len(self.timestamps), dtype=bool)

    def init(self):
        self.reference = get_reference_timestamps(self.timestamps)

    def __len__(self):
        return len(self.timestamps[self.reference]) - self.sample_size + 1

    def get_sample(self, index: int) -> Dict[str, int]:
        """"""
        if index > len(self) - self.sample_size:
            raise IndexError("Index out of range")

        sample = {key: [] for key in self.keys}
        for i in range(index, index + self.sample_size):
            for key in self.keys:
                sample[key].append(self.timestamps[key][i])
                self._updated[i] = True
        return sample
