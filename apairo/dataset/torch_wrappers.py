from torch.utils.data import Dataset, IterableDataset

from apairo.dataset.kitti import KittiDataset
from apairo.dataset.tartan_pt import TartanPT


class TorchKittiDataset(KittiDataset, Dataset):
    r"""Map-style PyTorch Dataset wrapping KittiDataset."""


class TorchKittiIterDataset(KittiDataset, IterableDataset):
    r"""Iterable-style PyTorch Dataset wrapping KittiDataset."""


class TorchTartanPTDataset(TartanPT, Dataset):
    r"""Map-style PyTorch Dataset wrapping TartanPT."""
