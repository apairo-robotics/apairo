"""
Dans le cas ou l'on voudrait separer les donnes de tests et d'entrainement dans un meme folder de donnees.
Pas sure de garder ceci.
"""
from abc import ABC, abstractmethod
from ..core import AbstractDataset


class DatasetSplitter(AbstractDataset, ABC):
    @abstractmethod
    def __getitem__(self):
        ...

        # class PercentSeparationDataset(DatasetSplitter):
        #     """A dataset that will be separate by differents percents."""

        #     def __init__(self, dataset, *split_percents):
        #         self.dataset = dataset
        #         assert 0 < sum(split_percents) <= 1.0, "Splitting percents is not logical"
        #         self.split_percents = split_percents
        #         self.lengths = list(map(lambda x: int(x*len(dataset)), self.split_percents))
        #         # Exemple : split_percents [0.8, 0.2] => lengths [80, 20] (for len(dataset) = 100)
        #         self.indexes = [0]
        #         for length in self.lengths: self.indexes.append(self.indexes[-1] + length)
        #         # indexes [0, 80, 100]

        #     def __getitem__(self, dataset_index, index):
        #         assert 0 <= index < self.lengths[dataset_index], "Index out of range"
        #         return self.dataset[self.indexes[dataset_index] + index]

        # class IndexSeparationDataset(DatasetSplitter):
        #     """A dataset that will be separate in two."""

        #     def __init__(self, dataset, *split_indexes):
        #         self.dataset = dataset
        #         assert 0 < max(split_indexes) < len(dataset), "Splitting indexes exceed dataset length"
        #         self.split_indexes = split_indexes
        #         # Exemple : split_percents [0.8, 0.2] => lengths [80, 20] (for len(dataset) = 100)
        #         self.indexes = [0]
        #         for length in self.lengths: self.indexes.append(self.indexes[-1] + length)
        #         # indexes [0, 80, 100]

        #     def __getitem__(self, dataset_index, index):
        #         assert 0 <= index < self.lengths[dataset_index], "Index out of range"
        #         return self.dataset[self.indexes[dataset_index] + index]
