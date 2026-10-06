"""Frame-naming policy for per-frame channels.

One source of truth, shared by the per-frame loader (which files it *reads*) and
the channel writer (which names it is allowed to *emit*).  A per-frame channel
stores one data file per frame; the stem (filename without extension) identifies
the frame.  ``_`` is reserved for suffixed sub-channel variants
(``000000_intensity.npy`` belongs to a separate ``intensity`` channel), so a
frame file's stem must not contain it -- the loader skips such files, and the
writer refuses to create them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apairo.core.formats import Format


def frame_stem_is_valid(stem: str) -> bool:
    """A per-frame file's stem must not contain ``_`` (reserved for suffixed
    sub-channel variants like ``000000_intensity``)."""
    return "_" not in stem


def is_frame_file(name: str, ext: str = ".npy") -> bool:
    """True if *name* is a per-frame data file the loader reads for this channel:
    the right extension and no sub-channel suffix."""
    return name.endswith(ext) and frame_stem_is_valid(Path(name).stem)


def suffixed_frame_files(directory, suffix: str, ext: str = ".npy") -> list[str]:
    """Frame-ordered files whose stem is ``<frame_stem>_<suffix>`` in *directory*.

    The counterpart of :func:`is_frame_file` for a suffixed sub-channel: instead
    of skipping ``000000_intensity.npy``, this lists exactly those files (for a
    given *suffix*), sorted the same way the legacy default sorts unsuffixed
    frames."""
    tail = f"_{suffix}{ext}"
    return sorted(
        f
        for f in os.listdir(directory)
        if f.endswith(tail) and frame_stem_is_valid(f[: -len(tail)])
    )


def require_frame_stem(stem: str) -> str:
    """Validate a frame stem the writer is about to emit; return it unchanged.

    Raises ``ValueError`` if the stem is empty, holds a path separator, or
    contains ``_`` (which the per-frame loader would skip -- the silent failure
    this policy exists to prevent)."""
    if not stem:
        raise ValueError("frame stem must be non-empty")
    if "/" in stem or os.sep in stem:
        raise ValueError(f"frame stem {stem!r} must not contain a path separator")
    if not frame_stem_is_valid(stem):
        raise ValueError(
            f"frame stem {stem!r} must not contain '_': the per-frame loader "
            f"reserves '_' for suffixed sub-channel variants (e.g. "
            f"000000_intensity.npy) and would skip this file."
        )
    return stem


def channel_frame_files(
    fmt: Format, directory: str | Path, meta: dict, *, label: str = "channel"
) -> list[str] | None:
    """The frame filenames the core resolves for a channel, before its format's
    own listing -- shared by loading and ``apairo status``, so both read the
    same frames:

    - an ``order`` (else a ``key: {name: ...}``) regex: the format's data files
      whose stem matches, sorted by the numeric value of the regex's first
      capture group (else by name). This is the ``order`` contract -- it lets a
      channel whose names carry a '_' (a Rellis ``<epoch>_<ms>``, which the
      default convention reserves for suffixes) enumerate anyway, filters out
      strays, and orders non-zero-padded indices correctly;
    - a ``suffix``: the variant's own files (``000000_intensity.npy``);
    - otherwise ``None``: the format lists its own frames.
    """
    import re

    order, key = meta.get("order"), meta.get("key")
    spec = (
        order
        if order is not None
        else key
        if isinstance(key, dict) and "name" in key
        else None
    )
    if spec is None:
        suffix = meta.get("suffix")
        if not suffix:
            return None
        ext = sorted(fmt.extensions)[0] if fmt.extensions else ""
        return suffixed_frame_files(directory, str(suffix), ext=ext)
    if not fmt.per_frame:
        from apairo.core.formats import formats

        per_frame = ", ".join(f.name for f in formats() if f.per_frame)
        raise ValueError(
            f"{label} declares a filename key/order but its loader '{fmt.name}' has "
            f"no per-frame files -- filename keys/order need a per-frame loader "
            f"({per_frame})."
        )
    pattern = spec.get("name") if isinstance(spec, dict) else None
    if pattern is None:
        raise ValueError(
            f"{label} needs an 'order' or 'key' regex ('name') to enumerate by; "
            f"got {spec!r}."
        )
    regex = re.compile(pattern)

    def order_key(name: str) -> tuple[int, str]:
        match = regex.search(Path(name).stem)
        first = match.groups()[0] if (match and match.groups()) else None
        return (int(first) if (first and first.isdigit()) else 0, name)

    names = sorted(
        (p.name for p in fmt.data_files(Path(directory)) if regex.search(p.stem)),
        key=order_key,
    )
    if not names:
        raise FileNotFoundError(
            f"{label}: no files in '{directory}' match the enumeration regex "
            f"{pattern!r}."
        )
    return names
