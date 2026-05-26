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


def _to_numpy(data) -> np.ndarray:
    if hasattr(data, "detach"):          # torch.Tensor
        return data.detach().cpu().numpy()
    return np.asarray(data)


def run(
    preprocessor: Preprocessor,
    dataset_cls: type,
    sequence_dir: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Run a preprocessor on a sequence and persist the output.

    Handles everything downstream of the user's :meth:`~Preprocessor.process`
    call: directory creation, file naming, timestamp writing, and
    :func:`~apairo.core.config.register_channel`.

    Args:
        preprocessor: A :class:`~apairo.core.preprocessor.FramePreprocessor`
            or :class:`~apairo.core.preprocessor.SequencePreprocessor` instance.
        dataset_cls: The dataset class to use for loading input data (must be
            a :class:`~apairo.core.configurable_dataset.ConfigurableDataset`
            subclass so that :meth:`register_channel` is available).
        sequence_dir: Path to the sequence directory.
        overwrite: If ``False`` (default) and the output directory already
            exists, raise :exc:`FileExistsError`.  Set to ``True`` to
            recompute and overwrite.

    Raises:
        FileExistsError: If the output directory exists and ``overwrite`` is
            ``False``.
        TypeError: If ``preprocessor`` is neither a :class:`FramePreprocessor`
            nor a :class:`SequencePreprocessor`.
    """
    sequence_dir = Path(sequence_dir)
    output_dir = sequence_dir / preprocessor.output_key

    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory '{output_dir}' already exists. "
            f"Pass overwrite=True to recompute."
        )
    output_dir.mkdir(exist_ok=True)

    dataset = dataset_cls(sequence_dir, keys=preprocessor.input_keys)
    n = len(dataset)
    logger.info(
        "%-20s  %s  (%d frame%s)",
        preprocessor.__class__.__name__,
        sequence_dir.name,
        n,
        "s" if n != 1 else "",
    )

    if isinstance(preprocessor, FramePreprocessor):
        _run_frame(preprocessor, dataset, output_dir)
    elif isinstance(preprocessor, SequencePreprocessor):
        _run_sequence(preprocessor, dataset, output_dir)
    else:
        raise TypeError(
            f"preprocessor must be a FramePreprocessor or SequencePreprocessor, "
            f"got {type(preprocessor).__name__}."
        )

    logger.info("Done  →  %s", output_dir)
    dataset_cls.register_channel(
        sequence_dir,
        preprocessor.output_key,
        preprocessor.output_loader,
        timestamps_from=preprocessor.timestamps_from,
        sources=preprocessor.sources,
    )


def _run_frame(preprocessor: FramePreprocessor, dataset, output_dir: Path) -> None:
    timestamps = []
    n = len(dataset)

    for idx, sample in enumerate(dataset):
        logger.debug("[%d/%d]", idx + 1, n)
        result = _to_numpy(preprocessor.process(sample))

        writer = WRITERS[preprocessor.output_loader]()
        ext = "npy" if preprocessor.output_loader in ("npys", "npys_img") else preprocessor.output_loader
        writer.write(result, output_dir / f"{idx:06d}.{ext}")

        if preprocessor.timestamps_from is None:
            timestamps.append(sample.timestamp)

    if timestamps:
        np.savetxt(output_dir / "timestamps.txt", timestamps)


def _run_sequence(preprocessor: SequencePreprocessor, dataset, output_dir: Path) -> None:
    result = _to_numpy(preprocessor.process(iter(dataset)))

    writer = WRITERS[preprocessor.output_loader]()
    writer.write(result, output_dir / f"{preprocessor.output_key}.{preprocessor.output_loader}")

    if preprocessor.timestamps_from is None:
        key = preprocessor.input_keys[0]
        np.savetxt(output_dir / "timestamps.txt", dataset.timestamps[key])
