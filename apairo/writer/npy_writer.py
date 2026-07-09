from pathlib import Path

import numpy as np


class NPYWriter:
    def write(self, arr: np.ndarray, path: Path) -> None:
        """Save arr to path as .npy. Creates parent dirs."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)
