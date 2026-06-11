from apairo.core import AbstractDataset
from apairo.core.sample import Sample


class MockDataset(AbstractDataset):
    """Minimal dataset honouring the AbstractDataset contract:
    ``_load(idx)`` returns a :class:`Sample` with one entry per active key."""

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

    def _load(self, idx: int) -> Sample:
        return Sample(data={k: self.data[k][idx] for k in self.keys if k in self.data})


def create_mock_dataset():
    return MockDataset()
