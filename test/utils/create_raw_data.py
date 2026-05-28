import os
import numpy as np
from test.paths import tmp_path


def create_timestamps_file(directory, start, end, n, freq_var=0.0, overwrite=False):
    if (
        os.path.isfile(os.path.join(tmp_path, directory, "timestamps.txt"))
        and not overwrite
    ):
        return
    mean = (end - start) / n
    timestamps = np.random.normal(mean, freq_var, n)
    timestamps = np.cumsum(timestamps)
    if not os.path.isdir(os.path.join(tmp_path, directory)):
        os.makedirs(os.path.join(tmp_path, directory), exist_ok=True)
    if tmp_path not in directory:
        directory = os.path.join(tmp_path, directory)
    file_path = os.path.join(directory, "timestamps.txt")
    with open(file_path, "w") as f:
        for t in timestamps:
            f.write(f"{t}\n")


def create_npy_file(data, filename, directory="", overwrite=False):
    if tmp_path not in directory:
        directory = os.path.join(tmp_path, directory)
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    elif os.path.isfile(os.path.join(directory, filename)) and not overwrite:
        return
    file_path = os.path.join(directory, filename)
    np.save(file_path, data)


def create_random_npy_file(len_, shape, filename, directory="", overwrite=False):
    if isinstance(shape, int):
        data = np.random.rand(len_, shape)
    else:
        data = np.random.rand(len_, *shape)
    create_npy_file(data, filename, directory, overwrite)


def create_random_npy_files(
    n_files, shape, directory="", file_spec="", overwrite=False
):
    for i in range(n_files):
        data = np.random.rand(*shape)
        filename = f"{i:06}_{file_spec}.npy" if file_spec else f"{i:06}.npy"
        create_npy_file(data, filename, directory, overwrite)


def create_random_images(n_images, shape, directory="", overwrite=False):
    from PIL import Image

    if tmp_path not in directory:
        directory = os.path.join(tmp_path, directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i in range(n_images):
        image = np.random.randint(0, 255, (*shape, 3), dtype=np.uint8)
        file_path = os.path.join(directory, f"{i:06d}.png")
        Image.fromarray(image, "RGB").save(file_path)
