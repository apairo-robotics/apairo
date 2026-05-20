
import numpy as np
import torch

from PIL import Image


def _png_to_numpy(file: str):
    """Convert a png file to a numpy array"""
    return np.array(Image.open(file))


def png_to_tensor(file: str, transform=None):
    """Convert a png filte to a tensor.
    A transform function can be applied to the tensor.
    With transform : numpy -> numpy
    """
    img = _png_to_numpy(file)
    if transform is not None:
        img = transform(img)
    return torch.from_numpy(img)


def png_rgb_to_tensor(file: str):
    """Convert a png file to a tensor"""
    def tranform(x): return np.transpose(x, (2, 0, 1)) / 255
    return png_to_tensor(file, transform=tranform)


def png_depth_to_tensor(file: str):
    """Convert a png file to a tensor"""
    def tranform(x): return np.expand_dims(x, axis=0) / 255
    return png_to_tensor(file, transform=tranform)


def conv2tensor(file: str):
    pass
