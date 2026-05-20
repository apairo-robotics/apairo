import os
import torch
from typing import Dict, Collection
import numpy as np

from src.utils import dict_flatten
from ..core import AbstractLoader
from ..core.utils.exceptions import FileExtensionError, EmptyLoaderWarning


class PTLoader(AbstractLoader):
    r"""A :class:`Loader` for `.pt` format.

    :attr:`pt` format is special because it can contains multiple features.
    In this case we consider the one from `Tartan Drive` dataset that contains at
    least the :attr:`dt` key that is the time between each frame.
    """
    data: Dict[str, torch.Tensor]
    lenght: int

    def __init__(self, file_path):
        self.file_path = file_path
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File path {file_path} not found or is not a file")
        extension = file_path.split(".")[-1]

        if not extension == "pt":
            raise FileExtensionError(f"This script extension {extension} is not pytorch file (pt)")
        _data = torch.load(file_path, weights_only=True)  # Weights only for safety mesure (see torch.load doc)

        self.data = dict_flatten(_data, format_key=lambda k, sk: "{}.{}".format(k, sk))
        self.lenght = len(self.data['dt'])
        for key in self.data.keys():
            assert self.lenght == self.data[key].shape[0], "Data size is not uniform"

    def set_keys(self, keys: Collection[str]):
        """Keep in the loader only the `keys`.

        Permit to free some space on the RAM.
        Generally, `dt` will not be in the keys so you may want to generate the timestamps beforehand
        with :meth:`get_timestamps`.
        """
        keys = set(keys)
        if 'dt' not in keys:
            keys.add('dt')
        current_keys = set(self.data.keys())
        for key in current_keys:
            if key not in keys:
                del self.data[key]
        if self.data.keys() == {'dt'}:
            raise EmptyLoaderWarning("No data left in the loader, reload the file to retrieve the data")

    def get_timestamps(self):
        return np.cumsum(self.data['dt'])

    def __len__(self):
        return self.lenght

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return {key: self.data[key][idx] for key in self.data.keys()}
        elif isinstance(idx, tuple):
            key, idx = idx
            return self.data[key][idx]
        else:
            raise TypeError(f"Index type {type(idx)} is not supported")

    @property
    def shape(self):
        return {
            key: self.data[key].shape[1:]
            for key in self.data.keys()
            if key != 'dt'
        }

    def reset(self):
        self.__init__(self.file_path)
