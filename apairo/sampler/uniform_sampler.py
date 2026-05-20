from ..core.abstract_sampler import AbstractSampler
from src.utils.timestamps import get_indexes, get_reference_timestamps


class LowFreqUniformSampler(AbstractSampler):
    """ Implement :class:`AbstractSampler` using frequency of the slowest data stream as reference.

    Samples the dataset by batching data according to the flow with the lowest frequency.
    For each iteration, the sampler stacks data corresponding to the slowest data stream.
    To be sure that each sample has the same size (because different batch can have different size),
    we will take for each key, the lowest frequency that can be found for batch creation as reference.
    """
    sample_size: int
    timestamps: dict
    is_uniform: bool = True

    def __init__(self, timestamps: dict, sample_size: int = 1):
        self.timestamps = timestamps
        self.sample_size = sample_size
        self.init()

    def init(self):
        self.init_timeline()

    def init_timeline(self):
        self.reference = get_reference_timestamps(self.timestamps)
        self.sample_last_indexes = get_indexes(self.timestamps, self.reference)
        self.sample_sizes = {
            key: min([self.sample_last_indexes[key][i + 1] - self.sample_last_indexes[key][i]
                     for i in range(len(self.sample_last_indexes[key]) - 1)])
            for key in self.keys
        }

    def get_sample(self, index):
        """Get the batch that correspond to the index in the reference timeline"""
        if index >= len(self):
            raise IndexError("Index out of range")
        return {
            {key: [self.sample_last_indexes[key][index] + i for i in range(self.sample_sizes[key])]} for key in self.keys
        }

    def __len__(self):
        return len(self.timestamps[self.reference]) // self.sample_size

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index >= len(self):
            raise StopIteration
        batch = [self.get_sample(self._index * self.sample_size + i) for i in range(self.sample_size)]
        self._index += 1
        return batch
