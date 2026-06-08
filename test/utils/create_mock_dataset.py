
from apairo.core import AbstractDataset


class MockDataset(AbstractDataset):
    def __init__(self, *args, **kwargs):
        self.data = {"key": ["value"]}
        self.keys = ["key"]

    def load(self, key: str, idx: int):
        return self.data[key][idx]

    def __len__(self):
        return 1

    @property
    def shape(self):
        return (1,)

    def __next__(self):
        return self.data

    def _load(self, idx):
        return self.data


def create_mock_dataset():
    return MockDataset()
