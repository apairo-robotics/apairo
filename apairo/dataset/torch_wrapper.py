from typing import overload, Dict, List, Tuple
from torch.utils.data import Dataset, IterableDataset

from .tartan_kitti import TartanKittiDataset
from .tartan_pt import TartanPT


class TorchTKDataset(TartanKittiDataset, Dataset):
    r"""A subclass :class:`TartanKittiDataset` used as a wrapper for :class:`torch.Dataset`.

    This is a map style dataset.
    This will be mainly used because we can use differents samplers with it and because
    we will put in it different ways of separating the dataset.

    Args:
        directory (str) :
            The directory where the dataset is stored
        keys (List) :
            The keys of the dataset

    """

    def __len__(self):
        return len(self.timeline)

    def __getitem__int(self, idx: int):
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range")
        # key, idx = self.timeline[idx]
        key = self.timeline[idx][0]
        idx = self.timeline[idx][1]
        return {"key": key, "data": self.loaders[key][idx], "timestamp": self.timestamps[key][idx]}

    def __getitem__tuple(self, idx: Tuple[str, int]) -> dict:
        key, idx = idx
        return {"key": key, "data": self.loaders[key][idx], "timestamp": self.timestamps[key][idx]}

    @overload
    def __getitem__(self, idx: int) -> dict:
        ...

    @overload
    def __getitem__(self, idx: Tuple[str, int]) -> dict:
        ...

    @overload
    def __getitem__(self, idx: Dict[str, List[int]]) -> dict:
        ...

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.__getitem__int(idx)
        elif isinstance(idx, Tuple):
            return self.__getitem__tuple(idx)
        else:
            raise TypeError(f"Index should be an int or a dict not {type(idx)}")


class TorchTKIterDataset(TartanKittiDataset, IterableDataset):
    r"""A subclass :class:`TartanKittiDataset` used as a wrapper for :class:`torch.IterableDataset`.

    This is a iterable style dataset. This will not be used in a first time but is kept because it follow
    the logic of model that train online.

    Args:
        directory (str) :
            The directory where the dataset is stored
        keys (List) :
            The keys of the dataset

    """

    def __next__(self):
        key, idx = super().__next__()
        return {"key": key, "data": self.loaders[key][idx], "timestamp": self.timestamps[key][idx]}


class TorchTPTDataset(TartanPT, Dataset):
    ...
