import numpy as np
from typing import List
from src.core import AbstractDataset


class ConcatDataset(AbstractDataset):
    r""" A :class:`AbstractRobotDataset` subclass that manage
    multiple dataset as one.


    ..note::
    It permits to concatenate multiple dataset of diferents types as well.
    The complexity here is to not load all the dataset at once
    but only the one that is needed.

    """

    datasets: List[AbstractDataset]
    keys: List[str]

    def __init__(self, datasets: List[AbstractDataset]):
        r""" Initialize the ConcatDataset.

        ..note::
        It will check if all the dataset have the same keys and set the keys.
        It will also set the lengths of each dataset and the cumulative lengths.
        The cumulative lengths is used to manage the index of the active dataset.
        """
        self.datasets = datasets
        self.check_keys(setter=True, warnings=False)

        self.lengths = np.array([len(dataset) for dataset in self.datasets], dtype=int)
        self.cumulative_lengths = np.cumsum(self.lengths)

    def _get_corresponding_dataset(self, idx) -> int|np.intp:
        """Get the corresponding dataset index of the given index."""
        if idx < 0 or idx >= self.cumulative_lengths[-1]:
            raise IndexError("Index out of range")
        return np.argmax(self.cumulative_lengths > idx)

    def check_keys(self, warnings=False, setter=True):
        """Check if all the dataset have the same keys.

        Args:
            warnings (bool) :
                A flag that indicates if the function should raise an error
                or a warning when the keys are not the same.
            setter (bool) :
                A flag that indicates if the keys should be set on each
                dataset to the same keys. (It will take the intersection of the keys)
        """
        keys = set(self.datasets[0].keys)
        for dataset in self.datasets:
            if keys != set(dataset.keys):
                if warnings:
                    raise KeyError("The keys of the dataset are not the same")
                else:
                    keys = keys.intersection(set(dataset.keys))
        if setter:
            for dataset in self.datasets:
                dataset.keys = keys
            self.keys = keys

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        """Get the item at the given index."""
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        dataset_idx = self._get_corresponding_dataset(idx)
        return self.datasets[dataset_idx][idx - self.cumulative_lengths[self.active_dataset]]

    def __iter__(self):
        self.active_dataset = 0
        return self

    def __next__(self):
        if self.active_dataset >= len(self.datasets):
            raise StopIteration
        return next(self.datasets[self.active_dataset])
