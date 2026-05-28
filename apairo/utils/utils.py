import os


def npy_analyser(folder):
    """Analyse the npy files in a folder and return the different formats.

    Example:
    livox
    |--- 000000.npy
    |--- 000000_intensity.npy
    |--- 000001.npy
    |--- ...

    >>> npy_analyser("livox")
    {"", "intensity"}

    """
    formats = set()
    for file in filter(lambda f: f[-3:] == "npy", os.listdir(folder)):
        if "_" not in file:
            formats.add("")
            continue
        file_ext = file.split("_")[-1].split(".")[0]
        formats.add(file_ext)
    return formats


def select_sequence(traj, keys, start, length):
    return {key: traj[key][start : start + length] for key in keys}


def dict_flatten(d, format_key=lambda _, sk: sk):
    """Take a tree of dict and flatten it.

    >>> dict_flatten({"a": {"b": 1}}, lambda k, sk: f"{k}.{sk}")
    {"a.b": 1}
    """
    flat_dict = {}
    for key, value in d.items():
        if isinstance(value, dict):
            flat_dict.update(
                {
                    format_key(key, subkey): subvalue
                    for subkey, subvalue in dict_flatten(value, format_key).items()
                }
            )
        else:
            flat_dict[key] = value
    return flat_dict


def map_recursive(x, func):
    if isinstance(x, dict):
        return {k: map_recursive(v, func) for k, v in x.items()}
    return func(x)
