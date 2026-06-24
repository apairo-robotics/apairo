"""Guard the example scripts so a doc/example drift fails CI instead of rotting.

A1-class bugs (an example referencing a channel that does not exist, a consumed
channel that is never created, ...) are only caught by *running* the file, not by
importing it. So every example is compile-checked, and the ones with a mini
fixture (Rellis-3D, TartanDrive) are executed end-to-end as subprocesses with the
dataset root injected via an environment variable.

Examples without a fixture (Goose / cross-dataset) are compile-checked only.
"""

import os
import py_compile
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parent.parent / "examples"
ASSETS = Path(__file__).parent / "assets"

ALL_EXAMPLES = sorted(p.name for p in EXAMPLES.glob("*.py"))

# Executed end-to-end against the mini fixtures (root injected via env var).
RELLIS_EXAMPLES = [
    "join_cached_prior.py",
    "rellis_traversability.py",
    "sequence_kfold.py",
    "training_pipeline_rellis.py",
]
TARTAN_EXAMPLES = [
    "tartan_frame_transform.py",
    "tartan_kitti_basic.py",
    "tartan_synchronize.py",
    "tartan_kitti_preprocess.py",
]


def _run(name: str, env: dict, cwd: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / name)],
        env={**os.environ, **env},
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"{name} exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


@pytest.mark.parametrize("name", ALL_EXAMPLES)
def test_example_compiles(name: str) -> None:
    """Every example must at least parse -- catches syntax rot in all of them."""
    py_compile.compile(str(EXAMPLES / name), doraise=True)


@pytest.mark.parametrize("name", TARTAN_EXAMPLES)
def test_tartan_example_runs(name: str, tmp_path: Path) -> None:
    seq = tmp_path / "figure_8"
    shutil.copytree(ASSETS / "mini_tartan" / "figure_8", seq)
    _run(name, {"APAIRO_TARTAN_SEQ": str(seq)}, cwd=tmp_path)


@pytest.mark.parametrize("name", RELLIS_EXAMPLES)
def test_rellis_example_runs(name: str, tmp_path: Path) -> None:
    root = tmp_path / "mini_rellis"
    shutil.copytree(ASSETS / "mini_rellis", root)
    _run(name, {"APAIRO_RELLIS_ROOT": str(root)}, cwd=tmp_path)
