# Installation

## Requirements

- Python ≥ 3.11
- PyTorch (CPU or CUDA)
- NumPy, torchvision, PyYAML (installed automatically)

## Install

```bash
pip install apairo
```

## Optional extras

```bash
# Visualization utilities (matplotlib)
pip install apairo[viz]

# Development tools (pytest, matplotlib)
pip install apairo[dev]
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
