from typing import Dict
import numpy as np

from src.utils.types import Timestamp
from src.utils.timestamps import get_indexes, get_reference_timestamps
from src.core import AbstractSampler


class LowFreqUniformSampler(AbstractSampler):
    r""" Implement :class:`AbstractSampler` using frequency of the slowest data stream as reference.

    Samples the dataset by batching data according to the flow with the lowest frequency.
    For each iteration, the sampler stacks data corresponding to the slowest data stream.
    To be sure that each sample has the same size (because different batch can have different size),
    we will take for each key, the lowest frequency that can be found for batch creation as reference.
    """
    is_uniform = True

    def __init__(self, timestamp: Timestamp, sample_size: int = 1, shuffle: bool = False):
        super().__init__(timestamp, sample_size=sample_size)
        self.shuffle = shuffle
        self.init()

    def __len__(self):
        return len(self.timestamps[self.reference]) - self.sample_size + 1

    def init(self):
        self.init_timeline()

    def init_timeline(self):
        self.reference = get_reference_timestamps(self.timestamps)
        self._indexes = np.array(range(len(self.timestamps[self.reference]) - self.sample_size + 1), dtype=int)
        self.sample_last_indexes = get_indexes(self.timestamps, self.reference)
        self._delta_indexes = {
            key: np.array(range(
                min([self.sample_last_indexes[key][i + 1] - self.sample_last_indexes[key][i]
                     for i in range(len(self.sample_last_indexes[key]) - 1)]) * self.sample_size),
                dtype=int)
            for key in self.keys
        }
        # print("INDEXES")
        # print(*[self.sample_last_indexes[key][:5] for key in self.keys])
        # print(*[self._delta_indexes[key][:5] for key in self.keys])

    def get_sample(self, index: int) -> Dict[str, int]:
        r"""Get the `sample` of indexes that correspond to the :arg:`index` in the dataset timeline.

        Be careful, this index is not the index of the sampler, but the index of the dataset.

        Example:
            If the dataset has 10 timestamps and the batch size is 3, the sampler will have 8 batches.
            (0, 1, 2), (1, 2, 3), (2, 3, 4), ... (7, 8, 9).
            If you want to get the first batch, you should call `get_sample(0)`.
            """
        if index > len(self) - 1:  # Fixed check to match new len
            raise IndexError("Index out of range")

        return {key: (index + self._delta_indexes[key]) for key in self.keys}

    def __iter__(self):
        self._index = 0
        if self.shuffle:
            np.random.shuffle(self._indexes)
        return self

    def __next__(self):
        if self._index >= len(self):
            raise StopIteration
        try:
            batch = self.get_sample(self._indexes[self._index])
        except IndexError:
            print(f"Index out of range: {self._index}")
            print(f"Length of the dataset: {len(self)}")
            raise StopIteration
        self._index += 1
        return batch

    def iterate(self, start=0, end=None):
        """Iterate over the dataset.
        """
        if end is None:
            end = len(self)
        for i in range(start, end):
            yield self.get_sample(i)
