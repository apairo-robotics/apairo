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

# Development tools (pytest, matplotlib)
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
```

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
