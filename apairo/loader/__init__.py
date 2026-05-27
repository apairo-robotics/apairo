from .img_loader import IMGLoader
from .npy_loader import NPYLoader
from .npys_loader import NPYSLoader
from .pt_loader import PTLoader
from .bin_loader import BINLoader
import os
from pathlib import Path
from typing import Callable
import numpy as np
import torch
import yaml
from torchvision.io import read_image


str_to_loader = {
    "img": IMGLoader,
    "npys": NPYSLoader,
    "npy": NPYLoader,
    "npys_img": NPYSLoader,
    "bin": BINLoader,
}

DERIVED_LOADERS: dict[str, Callable[[Path], torch.Tensor]] = {
    "npy": lambda path: torch.from_numpy(np.load(path)),
    "pt": lambda path: torch.load(path, weights_only=True),
    "bin": lambda path: torch.from_numpy(np.fromfile(path, dtype=np.float32)),
    "img": lambda path: read_image(str(path)),
}

__all__ = [
    "IMGLoader",
    "NPYLoader",
    "NPYSLoader",
    "PTLoader",
    "BINLoader",
    "str_to_loader",
    "DERIVED_LOADERS",
    "loads_timestamps",
    "load_profile",
]


def load_timestamps(file):
    return np.loadtxt(file)


def loads_timestamps(keys: list, files: dict) -> dict:
    r"""Load timestamps for each key from its subdirectory's ``timestamps.txt``.

    Keys without a ``timestamps.txt`` must be declared in ``timestamps_replacement``
    (TartanDrive legacy aliases) or handled upstream by the dataset class before
    calling this function.
    """
    timestamps_replacement = {
        "depth_left": "image_left",
        "local_dino_map": "local_gridmap",
        "stereo_colored_point_cloud_gmf": "stereo_colored_point_cloud",
    }
    timestamps = {}
    no_ts_dirs = []
    for key in keys:
        if key not in str_to_loader:
            if "timestamps.txt" in os.listdir(files[key]):
                timestamps[key] = load_timestamps(
                    os.path.join(files[key], "timestamps.txt")
                )
            else:
                no_ts_dirs.append(key)

    for key in no_ts_dirs:
        if key not in timestamps_replacement:
            raise ValueError(
                f"No timestamps.txt for '{key}' and no alias declared. "
                f"If this is a preprocessed channel, declare it via register_channel(..., timestamps_from=...)."
            )
        timestamps[key] = timestamps[timestamps_replacement[key]]

    return timestamps


def load_profile(profile_path: str | Path) -> dict:
    """Load a YAML loader-profile file.

    Each dataset subpackage bundles its own profile alongside its class and
    passes the absolute path here via ``Path(__file__).parent / 'profile.yaml'``.

    Args:
        profile_path: Absolute path to a YAML profile file.
    """
    with open(profile_path, "r") as f:
        return yaml.safe_load(f)
