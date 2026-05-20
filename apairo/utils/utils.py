import os
import torch
import numpy as np


def quaternion_invert(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Standard quaternion inversion.
    Args:
        quaternion: Tensor of shape (..., 4), where the last dimension is (w, x, y, z).
    Returns:
        Tensor of shape (..., 4) representing the inverse quaternion.
    """
    scaling = torch.tensor([1, -1, -1, -1], device=quaternion.device, dtype=quaternion.dtype)
    return quaternion * scaling


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions representing rotations.
    The convention is that q1 is applied after q2 (q1 * q2).
    Args:
        q1: Tensor of shape (..., 4) (w, x, y, z)
        q2: Tensor of shape (..., 4) (w, x, y, z)
    Returns:
        Tensor of shape (..., 4) representing q1 * q2
    """
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by dropping the last row.
    Args:
        matrix: batch of rotation matrices of size (..., 3, 3).
    Returns:
        6D rotation representation, of size (..., 6).
    """
    batch_dim = matrix.shape[:-2]
    return matrix[..., :2].clone().reshape(batch_dim + (6,))


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
    return {
        key: traj[key][start: start + length] for key in keys
    }


def get_rel_quat(quat, reference_quat):
    """Return the relative rotations in quaternion between two sequences of quaternions"""
    inv_quat = quaternion_invert(reference_quat)
    return quaternion_multiply(quat, inv_quat)


def quat_to_6drot(quat):
    return matrix_to_rotation_6d(quaternion_to_matrix(quat))


def dict_to_tensor(tensor: torch.Tensor, flatten_temporal_dim):
    if flatten_temporal_dim:
        return torch.cat([value.flatten(start_dim=1) for value in tensor.values()], dim=-1)
    else:
        return torch.cat([value.flatten(start_dim=2) for value in tensor.values()], dim=-1)


def tensor_to_dict(tensor: torch.Tensor, keys, dict_config):
    list_len = [np.prod(size) for key, size in dict_config if key in keys]
    tensor_split = torch.spit(tensor, list_len, dim=-1)

    return {
        key: tensor for key, tensor in zip(keys, tensor_split)
    }


def dict_flatten(d, format_key=lambda _, sk: sk):
    """Take a tree of dict and flatten it.
    The key_format function is used to format the key of the subdicts.
    An example of the key_format function is:
        format_key = lambda k, sk: f"{k}.{sk}"
    By default, the key_format function is the identity function with the subkey as the key.

    Args:
        d (dict): A tree of dict
        format_key (callable): A function that takes two arguments, the key and the subkey, and returns the new key.
    """
    flat_dict = {}
    for key, value in d.items():
        if isinstance(value, dict):
            flat_dict.update({format_key(key, subkey): subvalue for subkey,
                             subvalue in dict_flatten(value, format_key).items()})
        else:
            flat_dict[key] = value
    return flat_dict


def map_recursive(x, func):
    if isinstance(x, dict):
        return {k: map_recursive(v, func) for k, v in x.items()}
    return func(x)
