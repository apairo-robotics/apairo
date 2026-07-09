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
