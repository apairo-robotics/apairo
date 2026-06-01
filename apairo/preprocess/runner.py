from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING
import logging
import numpy as np

from apairo.core.preprocessor import FramePreprocessor, SequencePreprocessor
from apairo.writer import WRITERS

if TYPE_CHECKING:
    from apairo.core.preprocessor import Preprocessor

logger = logging.getLogger(__name__)

_LOADER_TO_EXT = {
    "npys": "npy",
    "npys_img": "npy",
    "npy": "npy",
    "bin": "bin",

}


def _to_numpy(data) -> np.ndarray:
    if hasattr(data, "detach"):  # torch.Tensor
        return data.detach().cpu().numpy()
    return np.asarray(data)


def run(
    preprocessor: Preprocessor,
    dataset_cls: type,
    root_dir: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Run a preprocessor on a dataset and persist the output channel.

    Uses ``dataset.derived_path()`` to determine where each output file is
    written, so every dataset can control its own file layout.  Registration
    is written to ``root_dir/.apairo``.

    Args:
        preprocessor: A :class:`~apairo.core.preprocessor.FramePreprocessor`
            or :class:`~apairo.core.preprocessor.SequencePreprocessor` instance.
        dataset_cls: Dataset class whose ``derived_path()`` defines file placement.
        root_dir: Dataset root directory (passed to ``dataset_cls.__init__``).
        overwrite: If ``False`` (default) and the first output file already
            exists, raise :exc:`FileExistsError`.

    Raises:
        FileExistsError: If output already exists and ``overwrite`` is ``False``.
        TypeError: If ``preprocessor`` is neither ``FramePreprocessor`` nor
            ``SequencePreprocessor``.
    """
    root_dir = Path(root_dir)
    dataset = dataset_cls(root_dir, keys=preprocessor.input_keys)
    n = len(dataset)

    ext = _LOADER_TO_EXT[preprocessor.output_loader]

    first_path = dataset.derived_path(0, preprocessor.output_key, ext)
    if first_path.exists() and not overwrite:
        raise FileExistsError(
            f"Derived key '{preprocessor.output_key}' already exists "
            f"(e.g. {first_path}). Pass overwrite=True to recompute."
        )

    logger.info(
        "%-20s  %s  (%d frame%s)",
        preprocessor.__class__.__name__,
        root_dir.name,
        n,
        "s" if n != 1 else "",
    )

    if isinstance(preprocessor, FramePreprocessor):
        _run_frame(preprocessor, dataset, ext)
    elif isinstance(preprocessor, SequencePreprocessor):
        _run_sequence(preprocessor, dataset, ext)
    else:
        raise TypeError(
            f"preprocessor must be a FramePreprocessor or SequencePreprocessor, "
            f"got {type(preprocessor).__name__}."
        )

    logger.info("Done  ->  '%s' registered in %s", preprocessor.output_key, root_dir)
    dataset_cls.register_channel(
        root_dir,
        preprocessor.output_key,
        preprocessor.output_loader,
        timestamps_from=preprocessor.timestamps_from,
        sources=preprocessor.sources,
    )


def _run_frame(preprocessor: FramePreprocessor, dataset, ext: str) -> None:
    writer = WRITERS[preprocessor.output_loader]()
    n = len(dataset)
    seq_timestamps: dict[Path, list] = {}

    for idx, sample in enumerate(dataset):
        logger.debug("[%d/%d]", idx + 1, n)
        result = _to_numpy(preprocessor.process(sample))
        path = dataset.derived_path(idx, preprocessor.output_key, ext)
        writer.write(result, path)

        if preprocessor.timestamps_from is None and sample.timestamp is not None:
            seq_timestamps.setdefault(path.parent, []).append(sample.timestamp)

    for seq_dir, timestamps in seq_timestamps.items():
        np.savetxt(seq_dir / "timestamps.txt", timestamps)


def _run_sequence(preprocessor: SequencePreprocessor, dataset, ext: str) -> None:
    result = _to_numpy(preprocessor.process(iter(dataset)))
    out = (
        dataset.root_dir / preprocessor.output_key / f"{preprocessor.output_key}.{ext}"
    )
    WRITERS[preprocessor.output_loader]().write(result, out)

    if preprocessor.timestamps_from is None:
        key = preprocessor.input_keys[0]
        np.savetxt(out.parent / "timestamps.txt", dataset.timestamps[key])
