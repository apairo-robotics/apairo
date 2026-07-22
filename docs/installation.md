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
# Benchmark / plotting utilities (matplotlib)
pip install apairo[bench]

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
