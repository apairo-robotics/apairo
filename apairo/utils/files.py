import os
from typing import Dict


def check_keys(keys: list, files: list) -> bool:
    """Check if the keys exist in the dataset"""
    for key in keys:
        if key not in files:
            raise ValueError(f"Key {key} not found in files")


def check_files(files: list, profile: dict) -> bool:
    """Check if the file names are in the profile."""
    for file in files:
        if file not in profile:
            raise ValueError(f"File {file} not in profile")


def get_files(directory: str) -> Dict[str, str]:
    """Get the files in the directory.

    Args:
        directory (str) :
            The directory where the files are stored

    Returns:
        Dict[str, str] :
            The files in the directory associated with their path
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory {directory} not found")
    files = list(filter(
        lambda x: os.path.isdir(os.path.join(directory, x)) and not x.startswith('_'),
        os.listdir(directory)
    ))
    return {file: os.path.join(directory, file) for file in files}
