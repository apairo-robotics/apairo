# Installation

## Requirements

- Python ≥ 3.11
- NumPy, PyYAML (installed automatically)

## Install

```bash
pip install apairo
```

## Optional extras

```bash
# Image channels (Pillow)
pip install apairo[vision]

# Zarr channels (zarr)
pip install apairo[zarr]

# Benchmark / plotting utilities (matplotlib)
pip install apairo[bench]

# Development tools (pytest, Pillow, zarr, ruff, mypy)
pip install apairo[dev]
```

## Standard datasets

Every standard dataset ships with apairo: its profile or its declaration is in
the package. Its extra installs what reading it needs:

```bash
pip install apairo[tartan]          # TartanDrive v2
pip install apairo[rellis]          # Rellis-3D
pip install apairo[goose]           # GOOSE
pip install apairo[semantic-kitti]  # SemanticKITTI
pip install apairo[tum]             # TUM RGB-D (declaration: tum_rgbd)
pip install apairo[euroc]           # EuRoC MAV (euroc_vicon_room, euroc_machine_hall)
pip install apairo[kuka]            # KUKA LBR Med F/T and IMU logs (kuka_ft_imu)
```

## Ecosystem

apairo reads datasets. The tools around it are separate packages, maintained
in the same organisation and published on PyPI. apairo never imports them. A
tool that has a command adds it to the `apairo` CLI, and `apairo --help` lists
the installed ones under "ecosystem commands".

| Package | What it does | Command |
|---|---|---|
| [`apairo_ros`](https://github.com/apairo-robotics/apairo_ros) | ROS bags to an apairo dataset, with its `.apairo` sidecars, from an interactive TUI or flags; or read in place (`BagDataset`) | `apairo ros` |
| [`apairo_transform`](https://github.com/apairo-robotics/apairo_transform) | Ready-made [transforms](transform.md) and augmentations, and the interpolators for [`synchronize()`](synchronization.md#interpolating-continuous-channels) | — |
| [`apairo_preprocess`](https://github.com/apairo-robotics/apairo_preprocess) | Ready-made [preprocessors](preprocessing.md): lidar odometry, ground segmentation, lidar–camera projection, traversability ground truth | — |
| [`apairo_visu`](https://github.com/apairo-robotics/apairo_visu) | Viewers, each behind its own extra | `apairo visu`, `apairo studio`, `apairo rerun` |

Each one installs as an extra of apairo, or on its own:

```bash
pip install "apairo[ros]"          # = pip install apairo-ros
pip install "apairo[transform]"    # = pip install apairo-transform
pip install "apairo[preprocess]"   # = pip install apairo-preprocess (brings apairo-transform)
pip install "apairo[visu]"         # = pip install "apairo-visu[studio]"
pip install "apairo[ros,transform,visu]"
```

An extra pins the satellite to the release line known to work with this
version of apairo.

`apairo_visu` installs nothing heavy by itself: pick the viewer you want with
its extra, or several at once (`[studio,rerun]`).

```bash
# apairo visu: Open3D lidar viewer with labels
pip install "apairo-visu[open3d]"

# apairo studio: the pipeline graph of a dataset, in the browser
pip install "apairo-visu[studio]"

# apairo rerun: lidar, cameras and labels in the Rerun viewer
pip install "apairo-visu[rerun]"
```

Each tool's README holds its full documentation.

## From source

```bash
git clone https://github.com/apairo-robotics/apairo
cd apairo
pip install -e ".[dev]"
```

## Verify

```python
import apairo
print(apairo.__version__)
```
