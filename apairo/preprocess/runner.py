from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from apairo.core.formats import Format, find_format, formats, get_format
from apairo.core.preprocessor import (
    FramePreprocessor,
    SequencePreprocessor,
    as_output_dict,
)
from apairo.core.sample import Sample

if TYPE_CHECKING:
    from apairo.core.preprocessor import Preprocessor

logger = logging.getLogger(__name__)

# apairo's own array formats: they hold any numeric array, so an output no
# other choice can store is written in them -- one file per frame, or one array.
_PER_FRAME_DEFAULT = "npys"
_STACKED_DEFAULT = "npy"


def _writable(name: str, label: str) -> Format:
    fmt = get_format(name)
    if not fmt.writes:
        writable = ", ".join(f.name for f in formats() if f.writes)
        raise ValueError(
            f"{label}: the '{name}' format is read-only. Formats that write: "
            f"{writable}."
        )
    return fmt


def _ext(fmt: Format) -> str:
    """The extension ``derived_path`` places a frame file under; a stacked
    format only uses the directory."""
    return fmt.write_suffix.lstrip(".") or fmt.name


class _OutputFormats:
    """The format each output channel is written in, decided on its first value
    -- one frame, or a sequence's whole result:

    1. the run's ``output_format``, else the preprocessor's ``output_loader``
       -- a format that cannot hold the output is refused, by name;
    2. otherwise the format of the input channel, when it can hold the output:
       an image mask stays an image, a cloud stays a cloud;
    3. otherwise ``npys`` (one ``.npy`` per frame) for a frame preprocessor,
       ``npy`` (one stacked array) for a sequence preprocessor.

    A stacked format holds every frame of a sequence, so it is ruled out when
    the run selects frames within sequences (a split listing frames): refused
    if requested, skipped if chosen.
    """

    def __init__(
        self,
        preprocessor: Preprocessor,
        requested: str | None,
        input_format: str | None,
        *,
        stacked_ok: bool = True,
    ) -> None:
        self._label = type(preprocessor).__name__
        self._requested = _writable(requested, self._label) if requested else None
        self._input = find_format(input_format)
        self._stacked_ok = stacked_ok
        self._per_frame_default = get_format(_PER_FRAME_DEFAULT)
        self._default = get_format(
            _STACKED_DEFAULT
            if isinstance(preprocessor, SequencePreprocessor) and stacked_ok
            else _PER_FRAME_DEFAULT
        )
        self.chosen: dict[str, Format] = {}
        self.automatic: set[str] = set()

    @staticmethod
    def _fits(fmt: Format, value: np.ndarray, *, frame: bool) -> bool:
        # One frame against a stacked format is tried as a one-row channel; a
        # sequence's result against a per-frame format, as its first row.
        if frame:
            return fmt.can_write(value if fmt.per_frame else value[None])
        return len(value) > 0 and fmt.can_write(value[0] if fmt.per_frame else value)

    def choose_all(self, results: dict[str, np.ndarray], *, frame: bool) -> dict:
        """Every output's format, decided before any of them is written: a
        refusal leaves nothing half-written."""
        return {
            key: self.choose(key, value, frame=frame) for key, value in results.items()
        }

    def choose(self, key: str, value: np.ndarray, *, frame: bool) -> Format:
        if key in self.chosen:
            return self.chosen[key]
        if self._requested is not None:
            fmt = self._requested
            if not fmt.per_frame and not self._stacked_ok:
                raise ValueError(
                    f"{self._label}: output '{key}' cannot be stored in the "
                    f"'{fmt.name}' format here: a stacked array holds every frame "
                    f"of a sequence, and this run covers a selection of them (a "
                    f"split). Run on every frame, or choose a per-frame format."
                )
            if not self._fits(fmt, value, frame=frame):
                writable = ", ".join(f.name for f in formats() if f.writes)
                raise ValueError(
                    f"{self._label}: output '{key}' ({value.dtype}, shape "
                    f"{value.shape}{' per frame' if frame else ''}) cannot be "
                    f"stored in the '{fmt.name}' format. Choose another with "
                    f"output_format= or output_loader ({writable})."
                )
        else:
            self.automatic.add(key)
            candidate = self._input
            if (
                candidate is not None
                and candidate.writes
                and (candidate.per_frame or self._stacked_ok)
                and self._fits(candidate, value, frame=frame)
            ):
                fmt = candidate
            else:
                fmt = self._default
        logger.info("%-20s  '%s' written as %s", self._label, key, fmt.name)
        self.chosen[key] = fmt
        return fmt

    def per_frame_instead(self, key: str) -> Format:
        """A format chosen automatically that turns out unable to hold the
        output: fall back to one file per frame."""
        self.chosen[key] = self._per_frame_default
        logger.info(
            "%-20s  '%s' written as %s instead", self._label, key, _PER_FRAME_DEFAULT
        )
        return self._per_frame_default

    def name_of(self, key: str) -> str:
        fmt = self.chosen.get(key) or self._requested or self._default
        return fmt.name


def _output_targets(
    dataset: Any, keys: list[str], root_dir: Path, dataset_cls: type
) -> list[tuple[Path, Path, str]]:
    """``(channel directory, registry directory, key)`` of every output, in
    every sequence: the directory ``derived_path`` places its first frame in,
    and the directory whose ``.apairo`` registers it -- the sequence's for the
    asynchronous family, the dataset root's for a profiled dataset."""
    from apairo.dataset.async_layout.dataset import AsyncLayoutDataset

    groups = getattr(dataset, "_seq_groups", None) or {None: list(range(len(dataset)))}
    firsts = [indices[0] for indices in groups.values() if indices]
    per_sequence = issubclass(dataset_cls, AsyncLayoutDataset)
    targets = []
    for key in keys:
        for idx in firsts:
            directory = dataset.derived_path(idx, key, "npy").parent
            targets.append(
                (directory, directory.parent if per_sequence else root_dir, key)
            )
    return targets


def _declared_channels(registry: Path, declare: Any) -> dict:
    """The channels of *registry* as loading sees them: ``channels.yaml``
    overlaid with the declarations (the root's, the sequence's, ``declare=``)."""
    from apairo.core.config import (
        config_exists,
        declaration_exists,
        declaration_path,
        inherited_declaration,
        merge_declared_channels,
        read_config,
        read_declaration,
    )

    channels = (
        read_config(registry).get("channels", {}) if config_exists(registry) else {}
    )
    own = declaration_path(registry) if declaration_exists(registry) else None
    for decl in (inherited_declaration(registry), own, declare):
        if decl is None or not Path(decl).is_file():
            continue
        try:
            channels = merge_declared_channels(channels, read_declaration(decl))
        except ValueError:
            pass
    return channels


def _refuse_collision(key: str, channels: dict, registry: Path) -> None:
    """An output may replace an earlier output of the same name, never a raw
    channel, a suffixed or colocated sub-channel, or another channel's alias."""
    entry = channels.get(key)
    if isinstance(entry, dict) and entry.get("kind", "raw") != "preprocess":
        raise FileExistsError(
            f"'{key}' is a raw channel in {registry}: a preprocess output cannot "
            f"replace it. Choose another output_key."
        )
    for name, meta in channels.items():
        if isinstance(meta, dict) and meta.get("alias") == key and name != key:
            raise FileExistsError(
                f"'{key}' is the public name of channel '{name}' in {registry} (its "
                f"alias): a preprocess output cannot take it. Choose another "
                f"output_key."
            )


def _input_format(dataset: Any, preprocessor: Preprocessor) -> str | None:
    """The format the preprocessor's reference input is stored in -- its
    ``timestamps_from`` channel, else its first input -- when the dataset
    family stores channels in formats (the asynchronous one)."""
    inputs = list(preprocessor.input_keys)
    if not inputs:
        return None
    ref = (
        preprocessor.timestamps_from
        if preprocessor.timestamps_from in inputs
        else inputs[0]
    )
    target = dataset
    if getattr(dataset, "_is_root", False) and getattr(dataset, "sequences", None):
        target = dataset.sequences[0]
    channel_format = getattr(target, "channel_format", None)
    return channel_format(ref) if channel_format is not None else None


def _to_numpy(data) -> np.ndarray:
    if hasattr(data, "detach"):  # torch.Tensor
        return data.detach().cpu().numpy()
    return np.asarray(data)


def _outputs_to_numpy(preprocessor, result) -> dict[str, np.ndarray]:
    return {k: _to_numpy(v) for k, v in as_output_dict(preprocessor, result).items()}


def _recipe_key(preprocessor, output_format: str | None = None) -> str:
    """A content hash of the preprocessor's *declared* configuration -- the
    identity of what it produces. Only explicit, serializable attributes go in
    (class name, declared I/O, and scalar constructor params -- never code, never
    array-valued attributes), so an identical recipe hits and a declared-param
    change mints a fresh one."""
    scalar = (int, float, str, bool, type(None))
    spec = {
        "class": type(preprocessor).__qualname__,
        "input_keys": list(preprocessor.input_keys),
        "outputs": list(preprocessor.outputs),
        # Under its historical name, so a recipe recorded before formats were
        # pluggable still matches.
        "output_loader": output_format or getattr(preprocessor, "output_loader", None),
        "timestamps_from": preprocessor.timestamps_from,
        "sources": list(preprocessor.sources or []),
        "params": {
            k: v
            for k, v in sorted(vars(preprocessor).items())
            if not k.startswith("_") and isinstance(v, scalar)
        },
    }
    blob = json.dumps(spec, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class _SameClockFrames:
    """The input channels of an asynchronous dataset, zipped row for row.

    An asynchronous dataset iterates its interleaved event timeline, one channel
    per sample, so a preprocessor with several inputs never sees them together.
    When every input sits on the *same* clock -- identical timestamps, typically
    a derived channel and the channel it was derived from -- interleaving them is
    pure loss: row ``i`` of each is one moment. This view serves that moment as
    one sample, and places outputs by row, the way a single-input run does.

    It exposes only what :func:`run` and its helpers use.
    """

    def __init__(self, dataset: Any) -> None:
        self._is_root: bool = bool(getattr(dataset, "_is_root", False))
        self.sequences: list[Any] = dataset.sequences if self._is_root else [dataset]
        self._clocks = [
            np.asarray(seq.timestamps[seq.keys[0]], dtype=float)
            for seq in self.sequences
        ]
        self._starts = np.concatenate([[0], np.cumsum([len(c) for c in self._clocks])])

    def __len__(self) -> int:
        return int(self._starts[-1])

    def _locate(self, idx: int) -> tuple[int, int]:
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        seq_idx = int(np.searchsorted(self._starts[1:], idx, side="right"))
        return seq_idx, idx - int(self._starts[seq_idx])

    def __getitem__(self, idx: int) -> Sample:
        seq_idx, row = self._locate(idx)
        seq = self.sequences[seq_idx]
        return Sample(
            data={key: seq.loaders[key][row] for key in seq.keys},
            timestamp=float(self._clocks[seq_idx][row]),
        )

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    @property
    def _seq_groups(self) -> dict[str, list[int]] | None:
        if not self._is_root:
            return None
        return {
            seq.root_dir.name: list(range(int(a), int(b)))
            for seq, a, b in zip(
                self.sequences, self._starts[:-1], self._starts[1:], strict=True
            )
        }

    def derived_path(self, idx: int, key: str, ext: str) -> Path:
        seq_idx, row = self._locate(idx)
        return self.sequences[seq_idx].derived_path(row, key, ext)


def _group_same_clock(dataset: Any, preprocessor: Preprocessor) -> Any:
    """Zip a multi-input preprocessor's inputs when they share one clock.

    Synchronous datasets and single-input runs are returned unchanged. On an
    asynchronous dataset, inputs on different clocks are refused: pairing them
    is a synchronisation, with a method and a tolerance to choose, not something
    the runner can guess.
    """
    if dataset.is_synchronous or len(preprocessor.input_keys) < 2:
        return dataset
    sequences = dataset.sequences if getattr(dataset, "_is_root", False) else [dataset]
    for seq in sequences:
        clocks = {key: np.asarray(seq.timestamps[key], dtype=float) for key in seq.keys}
        first = clocks[seq.keys[0]]
        if all(
            c.shape == first.shape and np.array_equal(c, first) for c in clocks.values()
        ):
            continue
        counts = ", ".join(f"'{key}' ({len(c)} frames)" for key, c in clocks.items())
        raise ValueError(
            f"{type(preprocessor).__name__} needs its inputs together in one "
            f"sample, but in '{seq.root_dir.name}' they are on different clocks: "
            f"{counts}. Inputs that share a clock -- identical timestamps, as "
            f"with a channel derived from another through timestamps_from -- are "
            f"grouped automatically. Channels on different clocks have to be "
            f"synchronized first, and running a preprocess over a synchronized "
            f"view is not supported yet."
        )
    logger.info(
        "%-20s  inputs %s share one clock -- grouped row for row",
        preprocessor.__class__.__name__,
        ", ".join(preprocessor.input_keys),
    )
    return _SameClockFrames(dataset)


def run(
    preprocessor: Preprocessor,
    dataset_cls: type[Any],
    root_dir: str | Path,
    *,
    overwrite: bool = False,
    reuse: bool = False,
    output_format: str | None = None,
    **dataset_kwargs,
) -> None:
    """Run a preprocessor on a dataset and persist the output channel.

    Uses ``dataset.derived_path()`` to determine where each output file is
    written, so every dataset can control its own file layout.  Registration
    is written to ``root_dir/.apairo``.

    On an asynchronous dataset a preprocessor with several ``input_keys`` gets
    them in one sample when they share one clock (identical timestamps, as with
    a channel derived through ``timestamps_from``): they are zipped row for row,
    and the output is numbered and stamped by that clock.

    The output is written in *output_format* when given, else in the
    preprocessor's ``output_loader``, else in its input channel's format when
    that format can hold it, else in ``npys`` / ``npy`` -- see
    :class:`_OutputFormats`. Any registered format that writes is accepted, a
    plugin's included.

    Args:
        preprocessor: A :class:`~apairo.core.preprocessor.FramePreprocessor`
            or :class:`~apairo.core.preprocessor.SequencePreprocessor` instance.
        dataset_cls: Dataset class whose ``derived_path()`` defines file placement.
        root_dir: Dataset root directory (passed to ``dataset_cls.__init__``).
        overwrite: If ``False`` (default) and the output channel already holds
            data, raise :exc:`FileExistsError`.
        reuse: Recipe-addressed idempotency. When ``True``, an output already on
            disk whose registered recipe matches this preprocessor's declared
            config is left untouched (the run is a no-op); a *changed* recipe is
            regenerated. The recipe hashes only declared attributes (never code),
            so it catches a scalar-parameter change but not an edit to
            ``__call__`` or an array-valued attribute -- use ``overwrite=True``
            after changing the body.
        output_format: The format to write the output in, overriding the
            preprocessor's ``output_loader``.

    Raises:
        FileExistsError: If output already exists and neither ``overwrite`` nor
            ``reuse`` is set.
        TypeError: If ``preprocessor`` is neither ``FramePreprocessor`` nor
            ``SequencePreprocessor``.
        ValueError: If a multi-input preprocessor's inputs are on different
            clocks of an asynchronous dataset, or the output format is read-only
            or cannot hold the output.
    """
    root_dir = Path(root_dir)
    if not isinstance(preprocessor, (FramePreprocessor, SequencePreprocessor)):
        raise TypeError(
            f"preprocessor must be a FramePreprocessor or SequencePreprocessor, "
            f"got {type(preprocessor).__name__}."
        )
    requested = output_format or getattr(preprocessor, "output_loader", None)
    dataset = dataset_cls(root_dir, keys=preprocessor.input_keys, **dataset_kwargs)
    outputs = _OutputFormats(
        preprocessor,
        requested,
        _input_format(dataset, preprocessor),
        # A split listing frames within sequences selects some of each one's
        # frames: a stacked array, row i being frame i, could not hold them.
        stacked_ok=getattr(dataset, "_frame_filter", None) is None,
    )
    dataset = _group_same_clock(dataset, preprocessor)
    n = len(dataset)

    recipe = _recipe_key(preprocessor, requested)
    targets = _output_targets(dataset, preprocessor.outputs, root_dir, dataset_cls)
    registries = {
        registry: _declared_channels(registry, dataset_kwargs.get("declare"))
        for _, registry, _ in targets
    }
    for _, registry, key in targets:
        _refuse_collision(key, registries[registry], registry)

    def holds_data(directory: Path) -> bool:
        return directory.is_dir() and any(directory.iterdir())

    def registered(key: str, registry: Path) -> dict | None:
        entry = registries[registry].get(key)
        return entry if isinstance(entry, dict) else None

    # reuse: an output already on disk under an identical recipe is a no-op; a
    # changed recipe regenerates (the old output is stale). Without reuse, any
    # existing output still raises unless overwrite is set.
    if reuse and all(
        holds_data(d) and (registered(key, r) or {}).get("recipe") == recipe
        for d, r, key in targets
    ):
        logger.info(
            "%-20s  %s  up to date (recipe match) -- skipped",
            preprocessor.__class__.__name__,
            root_dir.name,
        )
        return
    for directory, registry, key in targets:
        if not holds_data(directory):
            continue
        if not (overwrite or reuse):
            raise FileExistsError(
                f"Derived key '{key}' already exists (in {directory}). Pass "
                f"overwrite=True to recompute, or reuse=True to skip when the "
                f"recipe is unchanged."
            )
        if registered(key, registry) is None:
            raise FileExistsError(
                f"{directory} holds data apairo did not write as '{key}': it is "
                f"left alone. Delete it, or choose another output_key."
            )
        # Recomputed: the old output goes first, so files of a previous format
        # cannot be read back with the new ones.
        shutil.rmtree(directory)

    logger.info(
        "%-20s  %s  (%d frame%s)",
        preprocessor.__class__.__name__,
        root_dir.name,
        n,
        "s" if n != 1 else "",
    )

    if isinstance(preprocessor, FramePreprocessor):
        _run_frame(preprocessor, dataset, outputs)
    else:
        _run_sequence(preprocessor, dataset, outputs)

    keys = preprocessor.outputs
    logger.info("Done  ->  '%s' registered in %s", "', '".join(keys), root_dir)
    # Files land per sequence (derived_path routes each frame to its sub-sequence),
    # so on a root the channel must be declared in *each* sequence's channels.yaml
    # -- a single root-level registration would be unloadable. A single sequence
    # (root_dir is the sequence dir) registers once, unchanged.
    for seq_dir in dict.fromkeys(registry for _, registry, _ in targets):
        for key in keys:
            dataset_cls.register_channel(
                seq_dir,
                key,
                outputs.name_of(key),
                timestamps_from=preprocessor.timestamps_from,
                sources=preprocessor.sources,
                recipe=recipe,
            )


def _run_frame(
    preprocessor: FramePreprocessor, dataset, outputs: _OutputFormats
) -> None:
    """One call per frame. A per-frame format gets each output as it comes; a
    stacked one gets each sequence's outputs stacked, once the run is over."""
    n = len(dataset)
    seq_timestamps: dict[Path, list] = {}
    stacks: dict[tuple[Path, str], tuple[Format, list[tuple[int, np.ndarray]]]] = {}

    for idx, sample in enumerate(dataset):
        logger.debug("[%d/%d]", idx + 1, n)
        results = _outputs_to_numpy(preprocessor, preprocessor(sample))
        chosen = outputs.choose_all(results, frame=True)
        for key, result in results.items():
            fmt = chosen[key]
            path = dataset.derived_path(idx, key, _ext(fmt))
            if fmt.per_frame:
                fmt.write_frame(path, result)
            else:
                stacks.setdefault((path.parent, key), (fmt, []))[1].append(
                    (idx, result)
                )
            if sample.timestamp is not None:
                seq_timestamps.setdefault(path.parent, []).append(sample.timestamp)

    for (directory, key), (fmt, rows) in stacks.items():
        if len({row.shape for _, row in rows}) == 1:
            fmt.write_channel(directory, key, np.stack([row for _, row in rows]))
        elif key in outputs.automatic:
            # Chosen on frame 0, the stacked format cannot hold frames of
            # several shapes; nothing is written yet, so write them per frame.
            per_frame = outputs.per_frame_instead(key)
            for idx, row in rows:
                per_frame.write_frame(
                    dataset.derived_path(idx, key, _ext(per_frame)), row
                )
        else:
            raise ValueError(
                f"{preprocessor.__class__.__name__}: output '{key}' changes shape "
                f"from frame to frame, so it cannot be stacked into one "
                f"'{fmt.name}' array; write it per frame (npys)."
            )
    for directory, timestamps in seq_timestamps.items():
        np.savetxt(directory / "timestamps.txt", timestamps)


def _run_sequence(
    preprocessor: SequencePreprocessor, dataset, outputs: _OutputFormats
) -> None:
    """One call per sequence -- so it never crosses a sequence boundary. A
    stacked format gets the result as one array in the sequence's channel
    directory (``<seq>/<key>/``); a per-frame format gets one file per row, the
    layout a FramePreprocessor produces, which is what lets a multi-sequence
    ProfiledDataset (e.g. Rellis) find it per sequence."""
    # Per-sequence groups of global frame indices; datasets that do not expose a
    # sequence structure are treated as one sequence.
    groups = getattr(dataset, "_seq_groups", None) or {None: list(range(len(dataset)))}

    for indices in groups.values():
        if not indices:
            continue
        frames = [dataset[i] for i in indices]
        results = _outputs_to_numpy(preprocessor, preprocessor(iter(frames)))
        # Per-sequence clock, taken from the frames themselves, so each
        # sequence's output gets its own timestamps.txt of the right length.
        # Empty for a clockless dataset -> nothing written.
        seq_ts = [s.timestamp for s in frames if s.timestamp is not None]
        chosen = outputs.choose_all(results, frame=False)
        for key, result in results.items():
            fmt = chosen[key]
            directory = dataset.derived_path(indices[0], key, _ext(fmt)).parent
            if fmt.per_frame:
                if len(result) != len(indices):
                    raise ValueError(
                        f"{preprocessor.__class__.__name__} returned {len(result)} "
                        f"rows for key '{key}' on a {len(indices)}-frame sequence; "
                        f"a sequence preprocessor written per frame ('{fmt.name}') "
                        f"must return one row per input frame."
                    )
                for row, idx in zip(result, indices, strict=True):
                    fmt.write_frame(
                        dataset.derived_path(idx, key, _ext(fmt)), _to_numpy(row)
                    )
                if seq_ts:
                    np.savetxt(directory / "timestamps.txt", seq_ts)
            else:
                fmt.write_channel(directory, key, result)
                if seq_ts and len(seq_ts) == len(result):
                    np.savetxt(directory / "timestamps.txt", seq_ts)
