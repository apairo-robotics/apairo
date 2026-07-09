"""Helpers fabricating raw channel data on disk. Callers pass the target
directory (typically under pytest's ``tmp_path``); it is created if absent."""

import os

import numpy as np


def create_timestamps_file(directory, start, end, n, freq_var=0.0):
    os.makedirs(directory, exist_ok=True)
    mean = (end - start) / n
    timestamps = np.cumsum(np.random.normal(mean, freq_var, n))
    with open(os.path.join(directory, "timestamps.txt"), "w") as f:
        for t in timestamps:
            f.write(f"{t}\n")


def create_npy_file(data, filename, directory):
    os.makedirs(directory, exist_ok=True)
    np.save(os.path.join(directory, filename), data)


def create_random_npy_file(len_, shape, filename, directory):
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    create_npy_file(np.random.rand(len_, *shape), filename, directory)


def create_random_npy_files(n_files, shape, directory, file_spec=""):
    for i in range(n_files):
        filename = f"{i:06}_{file_spec}.npy" if file_spec else f"{i:06}.npy"
        create_npy_file(np.random.rand(*shape), filename, directory)


def create_random_images(n_images, shape, directory):
    from PIL import Image

    os.makedirs(directory, exist_ok=True)
    for i in range(n_images):
        image = np.random.randint(0, 255, (*shape, 3), dtype=np.uint8)
        Image.fromarray(image, "RGB").save(os.path.join(directory, f"{i:06d}.png"))
