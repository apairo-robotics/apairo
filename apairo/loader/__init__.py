import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import yaml

from .bin_loader import BINLoader
from .img_loader import IMGLoader
from .npy_loader import NPYLoader
from .npys_loader import NPYSLoader
from .tar_loader import TarImageLoader
from .txt_loader import TXTLoader
from .zarr_loader import ZarrLoader

str_to_loader = {
    "img": IMGLoader,
    "npys": NPYSLoader,
    "npy": NPYLoader,
    "bin": BINLoader,
    "zarr": ZarrLoader,
}


def _load_img(path: Path) -> np.ndarray:
    try:
        from PIL import Image

        return np.array(Image.open(path))
    except ImportError as exc:
        raise ImportError(
            "Loading image files requires Pillow. Install it with: pip install Pillow"
        ) from exc


DERIVED_LOADERS: dict[str, Callable[[Path], np.ndarray]] = {
    "npy": lambda path: np.load(path),
    "bin": lambda path: np.fromfile(path, dtype=np.float32),
    "img": _load_img,
}

__all__ = [
    "IMGLoader",
    "NPYLoader",
    "NPYSLoader",
    "BINLoader",
    "TXTLoader",
    "ZarrLoader",
    "TarImageLoader",
    "str_to_loader",
    "DERIVED_LOADERS",
    "loads_timestamps",
    "load_profile",
]


def load_timestamps(file):
    # atleast_1d: a single-frame channel's timestamps.txt is one line, which
    # np.loadtxt returns as a 0-d scalar -- callers index/iterate it as a 1-d array.
    return np.atleast_1d(np.loadtxt(file))


def loads_timestamps(keys: list, files: dict) -> dict:
    r"""Load timestamps for each key from its subdirectory's ``timestamps.txt``."""
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
    """Load a YAML loader-profile file."""
    with open(profile_path) as f:
        return yaml.safe_load(f)
