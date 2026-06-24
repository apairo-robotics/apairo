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
    **dataset_kwargs,
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
    dataset = dataset_cls(root_dir, keys=preprocessor.input_keys, **dataset_kwargs)
    n = len(dataset)

    ext = _LOADER_TO_EXT[preprocessor.output_loader]

    # Per-frame output (any FramePreprocessor, or a SequencePreprocessor that
    # emits one row per frame via output_loader="npys") is placed per frame by
    # derived_path; a stacked SequencePreprocessor writes one file per sequence
    # in that frame's channel directory (<seq>/<key>/<key>.ext).
    if isinstance(preprocessor, SequencePreprocessor) and preprocessor.output_loader != "npys":
        first_path = (
            dataset.derived_path(0, preprocessor.output_key, ext).parent
            / f"{preprocessor.output_key}.{ext}"
        )
    else:
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

        if sample.timestamp is not None:
            seq_timestamps.setdefault(path.parent, []).append(sample.timestamp)

    for seq_dir, timestamps in seq_timestamps.items():
        np.savetxt(seq_dir / "timestamps.txt", timestamps)


def _run_sequence(preprocessor: SequencePreprocessor, dataset, ext: str) -> None:
    if preprocessor.output_loader == "npys":
        _run_sequence_per_frame(preprocessor, dataset, ext)
    else:
        _run_sequence_stacked(preprocessor, dataset, ext)


def _run_sequence_stacked(preprocessor: SequencePreprocessor, dataset, ext: str) -> None:
    """Stacked output (output_loader="npy"): one {key}.{ext} file per sequence.

    ``process()`` runs once per sequence and the result is written to that
    sequence's channel directory (``<seq>/<key>/<key>.ext``) via ``derived_path``,
    so a multi-sequence ProfiledDataset finds one stacked file per sequence.  For
    a single-sequence dataset (an MNT mission, an async sequence) this is the same
    file the previous single-stream path wrote.
    """
    writer = WRITERS[preprocessor.output_loader]()
    groups = getattr(dataset, "_seq_groups", None) or {None: list(range(len(dataset)))}
    parent_ts = getattr(dataset, "timestamps", None)

    for indices in groups.values():
        if not indices:
            continue
        frames = [dataset[i] for i in indices]
        result = _to_numpy(preprocessor.process(iter(frames)))
        out = (
            dataset.derived_path(indices[0], preprocessor.output_key, ext).parent
            / f"{preprocessor.output_key}.{ext}"
        )
        writer.write(result, out)
        # Synchronous datasets have no timestamps -- nothing to propagate.
        if isinstance(parent_ts, dict):
            ts_key = preprocessor.timestamps_from or preprocessor.input_keys[0]
            np.savetxt(out.parent / "timestamps.txt", parent_ts[ts_key])


def _run_sequence_per_frame(
    preprocessor: SequencePreprocessor, dataset, ext: str
) -> None:
    """Per-frame output (output_loader="npys") from a sequence-level computation.

    ``process()`` runs once per sequence -- so it never crosses a sequence
    boundary -- and its rows are written one file per frame via ``derived_path``,
    the same on-disk layout a FramePreprocessor produces.  That is what lets a
    multi-sequence ProfiledDataset (e.g. Rellis) load the result back: a single
    stacked file at the dataset root is invisible to per-sequence discovery.
    """
    writer = WRITERS[preprocessor.output_loader]()
    # Per-sequence groups of global frame indices; datasets that do not expose a
    # sequence structure are treated as one sequence.
    groups = getattr(dataset, "_seq_groups", None) or {None: list(range(len(dataset)))}

    for indices in groups.values():
        frames = [dataset[i] for i in indices]
        result = _to_numpy(preprocessor.process(iter(frames)))
        if len(result) != len(indices):
            raise ValueError(
                f"{preprocessor.__class__.__name__}.process returned {len(result)} "
                f"rows for a {len(indices)}-frame sequence; a per-frame "
                f"(output_loader='npys') sequence preprocessor must return one row "
                f"per input frame."
            )
        seq_timestamps: list = []
        last_path = None
        for row, idx, sample in zip(result, indices, frames):
            last_path = dataset.derived_path(idx, preprocessor.output_key, ext)
            writer.write(_to_numpy(row), last_path)
            if sample.timestamp is not None:
                seq_timestamps.append(sample.timestamp)
        if seq_timestamps and last_path is not None:
            np.savetxt(last_path.parent / "timestamps.txt", seq_timestamps)
