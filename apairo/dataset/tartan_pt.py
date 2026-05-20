
from typing import Sequence
from src.loader import PTLoader
from ..core import AbstractDataset


class TartanPT(AbstractDataset):
    r""" A :class:`AbstractRobotDataset` subclass that manage pt format from Tartan drive

    This will never be used like this but be applied to wrapper,
    that will convert them into dataset object from libraries.

    Args:
        file_path (str) :
            The path where the dataset file is stored
        keys (List) :
            The keys of the dataset

    """
    synchronous = True

    def __init__(self, file_path: str, keys: Sequence):
        self._file_path = file_path
        self.keys = keys
        self.loaders = PTLoader(file_path)
        self.timestamp = self.loaders.get_timestamps()
        self.loaders.set_keys(keys)

    @property
    def file_path(self):
        return self._file_path

    @property
    def keys(self):
        return self._keys

    @keys.setter
    def keys(self, keys):
        self.loaders = PTLoader(self.file_path)
        self.timestamp = self.loaders.get_timestamps()
        self.loaders.set_keys(keys)
        self._keys = keys

    def __len__(self):
        return len(self.timestamp)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return {key: {
                "key": key,
                "data": self.loaders[key][idx],
                "timestamp": self.timestamp[idx]}
                for key in self.keys}
        elif isinstance(idx, tuple):
            key, idx = idx
            return {"data": self.loaders[key][idx], "timestamp": self.timestamp[idx]}
        else:
            raise TypeError(f"Index type {type(idx)} is not supported")

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        return (key, self.idx) if (key := self.keys[self.idx]) else StopIteration
