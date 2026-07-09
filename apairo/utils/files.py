import os


def get_files(directory: str) -> dict[str, str]:
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
    files = list(
        filter(
            lambda x: (
                os.path.isdir(os.path.join(directory, x)) and not x.startswith("_")
            ),
            os.listdir(directory),
        )
    )
    return {file: os.path.join(directory, file) for file in files}
