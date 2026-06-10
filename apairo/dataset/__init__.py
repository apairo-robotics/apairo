from .kitti import KittiDataset
from .tartan_kitti import TartanKittiDataset
from .goose import Goose3DDataset
from .rellis import Rellis3DDataset
from .semantic_kitti import SemanticKittiDataset
from .concat import ConcatDataset
from .zip import ZipDataset
from .mnt import MNTDataset

from apairo.core.config import register_channel


def split_sequences(
    datasets: list,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> tuple[list, list, list]:
    """Split datasets into train/val/test at the sequence level.

    Splitting at sequence level avoids temporal leakage between splits.

    Args:
        datasets: Ordered list (one entry per recording session).
        ratios: (train, val, test) fractions, must sum to 1.0.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios):.4f}")
    n = len(datasets)
    i1 = int(n * ratios[0])
    i2 = i1 + int(n * ratios[1])
    return datasets[:i1], datasets[i1:i2], datasets[i2:]


__all__ = [
    "KittiDataset",
    "TartanKittiDataset",
    "Goose3DDataset",
    "Rellis3DDataset",
    "SemanticKittiDataset",
    "ConcatDataset",
    "ZipDataset",
    "MNTDataset",
    "register_channel",
    "split_sequences",
]
