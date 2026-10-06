"""Test helpers for code built on apairo.

:func:`check_format` is the conformance check of the format contract: a format
plugin calls it from its own test suite on a small sample channel, and learns
there -- not from a user's ``apairo status`` -- what it gets wrong::

    from apairo.testing import check_format
    from apairo_xyz import XyzFormat

    def test_xyz_follows_the_contract(tmp_path):
        write_three_frames(tmp_path)
        check_format(XyzFormat(), tmp_path)

The built-in formats pass the same check, in apairo's own tests.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import numpy as np

from apairo.core.abstract_loader import AbstractLoader
from apairo.core.formats import Format

__all__ = ["check_format"]

# The core's clock forms: a format provides others, never these.
_CORE_KEY_FIELDS = frozenset({"name", "file", "units", "scale"})


def check_format(
    fmt: Format | type[Format], directory: str | Path, meta: dict | None = None
) -> None:
    """Check *fmt* against the format contract on a sample channel stored in
    *directory* (at least one frame). *meta* is the channel entry the sample
    needs (``array_file``, ``fields``, a ``key`` of the format's own form), the
    ``loader`` field excepted.

    Raises ``AssertionError`` listing every breach, so one run shows them all.
    """
    if isinstance(fmt, type):
        fmt = fmt()
    directory = Path(directory)
    meta = {"loader": getattr(fmt, "name", None), **(meta or {})}
    problems: list[str] = []
    _check_attributes(fmt, problems)
    if not problems:  # the rest needs a well-formed format
        _check_on_disk(fmt, directory, meta, problems)
    if problems:
        name = getattr(fmt, "name", type(fmt).__name__)
        raise AssertionError(
            f"format '{name}' breaks the apairo format contract:\n"
            + "\n".join(f"  - {p}" for p in problems)
        )


def _check_attributes(fmt: Format, problems: list[str]) -> None:
    name = getattr(fmt, "name", None)
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        problems.append(f"`name` must be a plain identifier, got {name!r}")
    loader = getattr(fmt, "loader", None)
    if not (isinstance(loader, type) and issubclass(loader, AbstractLoader)):
        problems.append(f"`loader` must be an AbstractLoader subclass, got {loader!r}")
    for attr in ("extensions", "fields", "key_forms"):
        value = getattr(fmt, attr)
        if not isinstance(value, frozenset) or not all(
            isinstance(v, str) for v in value
        ):
            problems.append(f"`{attr}` must be a frozenset of strings, got {value!r}")
    if not problems:
        bad = sorted(
            e for e in fmt.extensions if not e.startswith(".") or e != e.lower()
        )
        if bad:
            problems.append(f"`extensions` are lower-case and start with '.': {bad}")
        reserved = sorted(fmt.key_forms & _CORE_KEY_FIELDS)
        if reserved:
            problems.append(
                f"`key_forms` may not claim the core's key fields {reserved}"
            )
        if fmt.key_forms and type(fmt).clock is Format.clock:
            problems.append(
                f"`key_forms` {sorted(fmt.key_forms)} needs `clock()` implemented"
            )
    if not isinstance(fmt.priority, int) or isinstance(fmt.priority, bool):
        problems.append(f"`priority` must be an int, got {fmt.priority!r}")
    for attr in ("per_frame", "one_channel_per_file", "suffixes"):
        if not isinstance(getattr(fmt, attr), bool):
            problems.append(f"`{attr}` must be a bool")


def _check_on_disk(
    fmt: Format, directory: Path, meta: dict, problems: list[str]
) -> None:
    # Discovery.
    try:
        files = fmt.data_files(directory)
    except Exception as exc:
        problems.append(f"data_files() raised {exc!r}")
        return
    if fmt.per_frame and not files:  # a stacked store may be the directory itself
        problems.append(f"data_files() finds no frame in the sample {directory}")
    for p in files:
        if (
            p.parent != directory
            or p.name.startswith(".")
            or p.name == "timestamps.txt"
        ):
            problems.append(f"data_files() returned {p}, not a data file of the sample")
    if not _safe(fmt.detect, directory, problems):
        problems.append("detect() does not recognize the sample directory")
    with tempfile.TemporaryDirectory() as empty:
        if _safe(fmt.detect, Path(empty), problems):
            problems.append("detect() claims an empty directory")

    # Reading.
    try:
        loader = fmt.open(directory, meta, None)
    except Exception as exc:
        problems.append(f"open() raised {exc!r} on the sample")
        return
    n = len(loader)
    if n < 1:
        problems.append("the loader of the sample has no frames")
        return
    try:
        first = np.asarray(loader[0])
        np.asarray(loader[n - 1])
    except Exception as exc:
        problems.append(f"reading a frame raised {exc!r}")
        return
    if fmt.per_frame:
        names = getattr(loader, "files", None)
        if not (isinstance(names, list) and len(names) == n):
            problems.append(
                "a per-frame loader exposes `files`, one filename per frame (the "
                "core parses filename keys from it)"
            )
        else:
            try:
                one = fmt.open(directory, meta, names[:1])
            except Exception as exc:
                problems.append(f"open(files=[...]) raised {exc!r}")
            else:
                if len(one) != 1:
                    problems.append(
                        "open() ignores `files`: given one filename, the loader "
                        f"has {len(one)} frames"
                    )

    # Describing.
    try:
        facts = fmt.facts(directory, meta, None)
    except Exception as exc:
        problems.append(f"facts() raised {exc!r} on the sample")
    else:
        if facts.frames is not None and facts.frames != n:
            problems.append(f"facts() says {facts.frames} frames, the loader has {n}")
        if facts.shape is not None and list(facts.shape) != list(first.shape):
            problems.append(
                f"facts() says shape {facts.shape}, frame 0 has {list(first.shape)}"
            )
        if facts.dtype is not None and facts.dtype != str(first.dtype):
            problems.append(
                f"facts() says dtype {facts.dtype}, frame 0 is {first.dtype}"
            )
        if facts.clock is not None and len(facts.clock) != n:
            problems.append(
                f"facts() gives {len(facts.clock)} timestamps for {n} frames"
            )
    spec = meta.get("key")
    if isinstance(spec, dict) and set(spec) & fmt.key_forms:
        try:
            clock = np.asarray(fmt.clock(loader, spec, "sample"), dtype=float)
        except Exception as exc:
            problems.append(f"clock() raised {exc!r} on the sample's key")
        else:
            if clock.shape != (n,):
                problems.append(f"clock() gives shape {clock.shape} for {n} frames")
    hints = _safe(lambda: fmt.declare_hints(directory, meta), None, problems)
    if hints is not None:
        if not isinstance(hints, list) or not all(
            isinstance(h, str) and h.startswith("    ") for h in hints
        ):
            problems.append("declare_hints() returns lines indented four spaces")
    issues = _safe(lambda: fmt.validate("sample", meta, directory), None, problems)
    if issues is not None and (
        not isinstance(issues, list) or not all(isinstance(i, str) for i in issues)
    ):
        problems.append("validate() returns a list of issue strings")
    elif issues:
        problems.append(f"validate() rejects the sample: {issues}")


def _safe(call, arg, problems: list[str]):
    """``call(arg)`` (or ``call()``), recording an exception as a breach: the
    discovery and description methods are asked about any directory and must
    answer, not raise."""
    try:
        return call(arg) if arg is not None else call()
    except Exception as exc:
        problems.append(f"{getattr(call, '__name__', 'a method')}() raised {exc!r}")
        return None
