"""The conformance check of a dataset -- a standard one, or one a package adds.

:func:`check_dataset` opens a sample of the dataset the way a user would --
``init``, then loading -- and checks what every dataset owes its users, plus
what the sample's own description says to expect. It works on a stand-in of
the sample (directories recreated, files linked), so nothing is ever written
next to the data, and a full dataset can be checked as safely as a fixture.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

_COPY = "check_dataset_copy"


def check_dataset(
    dataset: type[Any] | str,
    root: str | Path,
    *,
    keys: list[str] | None = None,
    declare: str | Path | None = None,
    expect: dict[str, dict] | None = None,
    paired: list[list[str]] | None = None,
    clock: bool | None = None,
    splits: list[str] | None = None,
    frames: int | None = None,
    round_trip: bool = True,
) -> None:
    """Check a dataset on a sample of it, and list every breach.

    Every dataset is checked on:

    - ``init`` and loading, as ``apairo init --as <Class>`` and the class do;
    - each channel's first, middle and last frames: arrays whose dtype, rank
      and trailing dimensions do not change;
    - its clock: finite, one timestamp per frame, never going back within a
      sequence;
    - ``apairo check``, which must report nothing;
    - with *round_trip*, a preprocess output written into it and read back
      unchanged, which exercises where its layout places derived channels.

    Args:
        dataset: The dataset class, or its registered name.
        root: The sample: a dataset root or a sequence, as a user passes it.
        keys: Channels to load; ``None`` for the dataset's default.
        declare: A declaration (a file, or the name of a shipped one) the
            dataset is read through.
        expect: Per channel, what its frames are: ``shape`` (``None`` for a
            free dimension), ``dtype``, and a value ``range`` ``[low, high]``
            -- for the channels that are loaded.
        paired: Groups of channels with one row per element of the same frame
            (a cloud and its per-point labels).
        clock: ``True`` when the dataset must have a clock, ``False`` when it
            must not, ``None`` for either.
        splits: Splits that must load, each non-empty, none sharing a frame.
        frames: The number of frames the sample holds.
        round_trip: Write a preprocess output and read it back. Turn it off on
            a full dataset, whose copy of a channel would be large.

    Raises:
        AssertionError: listing every breach.
    """
    from apairo.dataset.registry import get_dataset, resolve_declaration

    cls = get_dataset(dataset) if isinstance(dataset, str) else dataset
    declared = resolve_declaration(declare)
    extra: dict[str, Any] = {"declare": declared} if declared is not None else {}
    problems: list[str] = []
    with tempfile.TemporaryDirectory() as tmp, _known(cls):
        work = _stand_in(Path(root).expanduser(), Path(tmp) / Path(root).name)
        try:
            cls.init(work, **extra)
            ds = cls(work, keys=keys, **extra)
        except Exception as exc:
            problems.append(f"init and loading failed: {exc!r}")
        else:
            _check_loaded(ds, keys, expect or {}, paired or [], frames, problems)
            _check_clock(ds, clock, problems)
            _check_apairo_check(work, declared, problems)
            if splits:
                _check_splits(cls, work, keys, extra, splits, problems)
            if round_trip:
                _check_round_trip(cls, work, ds, extra, problems)
    if problems:
        raise AssertionError(
            f"{cls.__name__} on {root} breaks the dataset contract:\n"
            + "\n".join(f"  - {p}" for p in problems)
        )


@contextmanager
def _known(cls: type[Any]) -> Iterator[None]:
    """*cls* registered for the length of the check, so that ``apairo check``
    reads the sample as that dataset -- a class a test defines, or a package's
    not installed yet, is not in the registry otherwise."""
    from apairo.dataset import registry

    registry.dataset_names()
    added = registry.find_dataset(cls.__name__) is None
    if added:
        registry.register_dataset(cls)
    try:
        yield
    finally:
        if added:
            registry._DATASETS.pop(cls.__name__, None)


def _stand_in(src: Path, dst: Path) -> Path:
    """A writable stand-in for *src*: its directories recreated, its files
    linked (copied where links are refused), any ``.apairo`` sidecar left out
    -- so the check starts from the data as downloaded and writes nothing next
    to it."""

    def link(source: str, target: str) -> None:
        try:
            os.symlink(os.path.abspath(source), target)
        except OSError:
            shutil.copy2(source, target)

    shutil.copytree(
        src, dst, copy_function=link, ignore=shutil.ignore_patterns(".apairo")
    )
    return dst


def _picks(n: int) -> list[int]:
    return sorted({0, n // 2, n - 1}) if n else []


def _check_loaded(ds, keys, expect, paired, frames, problems: list[str]) -> None:
    n = len(ds)
    if n == 0:
        problems.append("the dataset loads no frame")
        return
    if frames is not None and n != frames:
        problems.append(f"{n} frames, the sample holds {frames}")
    loaders = ds.loaders
    for key in keys or []:
        if key not in loaders:
            problems.append(
                f"channel '{key}' is not loaded (loaded: {sorted(loaders)})"
            )
    for key, loader in loaders.items():
        seen: tuple | None = None
        for i in _picks(len(loader)):
            try:
                arr = np.asarray(loader[i])
            except Exception as exc:
                problems.append(f"'{key}' frame {i} does not read: {exc!r}")
                break
            form = (arr.dtype, arr.ndim, arr.shape[1:])
            if seen is not None and form != seen:
                problems.append(
                    f"'{key}' frame {i} is {arr.dtype} {arr.shape}, frame 0 was "
                    f"{seen[0]} with trailing dimensions {seen[2]}"
                )
            seen = seen or form
            if key in expect:
                _check_expected(key, i, arr, expect[key], problems)
    for group in paired:
        present = [k for k in group if k in loaders]
        lengths = {len(loaders[k]) for k in present}
        if len(present) < 2 or len(lengths) > 1:
            continue
        for i in _picks(lengths.pop()):
            rows = {k: len(np.asarray(loaders[k][i])) for k in present}
            if len(set(rows.values())) > 1:
                problems.append(f"frame {i}: paired channels differ in length {rows}")


def _check_expected(key: str, i: int, arr: np.ndarray, spec: dict, problems) -> None:
    shape = spec.get("shape")
    if shape is not None and (
        arr.ndim != len(shape)
        or any(s is not None and s != d for s, d in zip(shape, arr.shape, strict=False))
    ):
        problems.append(f"'{key}' frame {i} has shape {arr.shape}, expected {shape}")
    dtype = spec.get("dtype")
    if dtype is not None and arr.dtype != np.dtype(dtype):
        problems.append(f"'{key}' frame {i} is {arr.dtype}, expected {dtype}")
    bounds = spec.get("range")
    if bounds is not None and arr.size:
        low, high = bounds
        if arr.min() < low or arr.max() > high:
            problems.append(
                f"'{key}' frame {i} holds values {arr.min()}..{arr.max()}, outside "
                f"[{low}, {high}]"
            )


def _check_clock(ds, clock: bool | None, problems: list[str]) -> None:
    stamps = ds.timestamps
    if isinstance(stamps, dict):  # asynchronous: one clock per channel
        clocks = {
            k: (np.asarray(v, dtype=float), len(ds.loaders[k]))
            for k, v in stamps.items()
        }
        groups = {k: [np.arange(len(c))] for k, (c, _) in clocks.items()}
    elif stamps is None:
        if clock:
            problems.append("no clock, and the dataset must have one")
        return
    else:  # synchronous: one shared clock, restarting with each sequence
        arr = np.asarray(stamps, dtype=float)
        clocks = {"the frame clock": (arr, len(ds))}
        ids = np.asarray(ds.frame_sequence_ids)
        groups = {
            "the frame clock": [np.flatnonzero(ids == s) for s in dict.fromkeys(ids)]
        }
    if clock is False:
        problems.append("a clock, and the dataset must have none")
    for name, (arr, n) in clocks.items():
        if len(arr) != n:
            problems.append(f"{name}: {len(arr)} timestamps for {n} frames")
            continue
        if not np.all(np.isfinite(arr)):
            problems.append(f"{name} holds non-finite timestamps")
        for rows in groups[name]:
            if np.any(np.diff(arr[rows]) < 0):
                problems.append(f"{name} goes back in time")
                break


def _check_apairo_check(work: Path, declared: Path | None, problems: list[str]) -> None:
    """What ``apairo check [--declare ...]`` reports: the dataset, and the
    external declaration validated against it."""
    from apairo.cli import _check_issues, _external_declare_issues

    issues = _check_issues(work, declared)
    if issues is None:
        problems.append("apairo check does not recognize it as a dataset")
        return
    if declared is not None:
        issues += _external_declare_issues(work, str(declared))
    problems.extend(f"apairo check: {issue}" for issue in issues)


def _check_splits(cls, work, keys, extra, splits, problems: list[str]) -> None:
    seen: dict[tuple[str, str], str] = {}
    for split in splits:
        try:
            part = cls(work, keys=keys, split=split, **extra)
        except Exception as exc:
            problems.append(f"split '{split}' does not load: {exc!r}")
            continue
        if len(part) == 0:
            problems.append(f"split '{split}' is empty")
        for frame in zip(
            map(str, part.frame_sequence_ids), map(str, part.frame_stems), strict=True
        ):
            if frame in seen:
                problems.append(
                    f"frame {frame} is in both '{seen[frame]}' and '{split}'"
                )
                break
            seen[frame] = split


def _check_round_trip(cls, work: Path, ds, extra, problems: list[str]) -> None:
    from apairo.core.preprocessor import FramePreprocessor

    ref = next(iter(ds.loaders))

    class _Copy(FramePreprocessor):
        output_key = _COPY
        input_keys = [ref]

        def __call__(self, sample):
            return sample.data[ref]

    try:
        cls.run_preprocess(_Copy(), work, output_format="npys", **extra)
        back = cls(work, keys=[ref, _COPY], **extra)
    except Exception as exc:
        problems.append(f"a preprocess output does not write and load back: {exc!r}")
        return
    original, copy = back.loaders[ref], back.loaders[_COPY]
    if len(copy) != len(original):
        problems.append(
            f"the preprocess output has {len(copy)} frames, '{ref}' {len(original)}"
        )
        return
    for i in _picks(len(original)):
        if not np.array_equal(np.asarray(copy[i]), np.asarray(original[i])):
            problems.append(f"the preprocess output's frame {i} differs from '{ref}'")
            break
