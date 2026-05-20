from .img_loader import IMGLoader
from .npy_loader import NPYLoader
from .npys_loader import NPYSLoader
from .pt_loader import PTLoader
import os
import numpy as np
import yaml
from src.utils.paths import dataset_profile_directory_path


str_to_loader = {
    "img": IMGLoader,
    "npys": NPYSLoader,
    "npy": NPYLoader,
    "npys_img": NPYSLoader  # For the moment, we will use npys by default
}

# Loader is a abstract class : not public
__all__ = [
    "Loader",
    "IMGLoader",
    "NPYLoader",
    "NPYSLoader",
    "PTLoader",
    "str_to_loader",
    "loads_timestamps",
    "load_profile"
]


def load_timestamps(file):
    return np.loadtxt(file)


def loads_timestamps(keys: list, files: dict) -> dict:
    r"""Load all the timestamps from the files.

    To do so, we will load all the timestamps files that exist on each directory
    then for the directory that do not have the timestamps, we will use the timestamps of the closest directory.
    It will return a dictionary with the timestamps of each fields.
    Args:
        files (list[str]) :
            List of the files in the dataset (they are presumed to be the directories)
    """
    timestamps_replacement = {
        "depth_left": "image_left",
        "local_dino_map": "local_gridmap",
        "stereo_colored_point_cloud_gmf": "stereo_colored_point_cloud",
    }
    timestamps = {}
    no_ts_directorys = []
    # Load the timestamps that already exist
    for key in keys:
        if key not in str_to_loader:
            if "timestamps.txt" in os.listdir(files[key]):
                timestamps[key] = load_timestamps(os.path.join(files[key], "timestamps.txt"))
            else:
                no_ts_directorys.append(key)

    # Load the timestamps that are not originally in the directory
    for no_ts_directory in no_ts_directorys:
        if no_ts_directory not in timestamps_replacement:
            raise ValueError(f"No timestamps for {no_ts_directory}")
        timestamps[no_ts_directory] = timestamps[timestamps_replacement[no_ts_directory]]

    return timestamps


def load_profile(name_file="profile.yaml"):
    """Load a `yaml` profile file. It is presumed to be in the `src/parameter` directory."""
    with open(os.path.join(dataset_profile_directory_path, name_file), 'r') as file:
        return yaml.load(file, Loader=yaml.FullLoader)
