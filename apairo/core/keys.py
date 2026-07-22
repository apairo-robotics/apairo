"""Parse a channel's per-frame alignment key (its *clock*) from the filenames it
already has, or from a named sidecar -- the form-agnostic mechanism shared by the
synchronous (:class:`~apairo.core.profiled_dataset.ProfiledDataset`) and
asynchronous (:class:`~apairo.dataset.async_layout.AsyncLayoutDataset`) families.

Pure: the only I/O is reading a named sidecar for the ``{file: ...}`` form;
nothing is ever written into the dataset tree.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from apairo.core.config import KEY_UNITS, safe_config_name


def parse_filename_key(
    files: list,
    spec: dict,
    *,
    directory: Path | str | None = None,
    label: str = "channel",
) -> np.ndarray:
    """One key per file, as a float array.

    ``spec`` is the channel's ``key`` mapping, in one of two forms:

    - ``{file: <name>}`` -- read the keys from a named sidecar in *directory*
      (one float per line, a differently-named ``timestamps.txt``);
    - ``{name: <regex>}`` (+ optional ``scale`` / ``units``) -- parse the key from
      each filename stem. Capture groups combine as ``sum(int(g_i) * scale_i)``
      with a ``scale``/``units``, else ``float('.'.join(groups))`` (one group is an
      index, two are ``<int>.<frac>``).

    *label* prefixes error messages (e.g. ``"Channel 'camera'"``).
    """
    if "file" in spec:
        from apairo.loader import load_timestamps

        if directory is None:
            raise ValueError(f"{label}: key {{file}} needs a directory to read from.")
        path = Path(directory) / safe_config_name(
            spec["file"], label=f"{label} key file"
        )
        if not path.exists():
            raise FileNotFoundError(
                f"{label}: key file '{spec['file']}' not found in '{directory}'."
            )
        return load_timestamps(path)

    pattern = spec.get("name")
    if pattern is None:
        raise ValueError(
            f"{label}: 'key' spec needs a 'name' regex or a 'file'; got {spec!r}."
        )
    regex = re.compile(pattern)
    scale = spec.get("scale")
    units = spec.get("units")
    if units is not None:
        # `units` is self-documenting sugar for `scale`: each capture group is a
        # time field (s / ms / us / ns) folded to seconds.
        if scale is not None:
            raise ValueError(
                f"{label}: key has both 'units' and 'scale' -- 'units' is sugar "
                f"for 'scale', give one."
            )
        try:
            scale = [KEY_UNITS[u] for u in units]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"{label}: unknown key unit in {units!r}; known: {sorted(KEY_UNITS)}."
            ) from exc
    if regex.groups == 0:
        raise ValueError(f"{label}: key regex {pattern!r} has no capture group.")
    if scale is not None and len(scale) != regex.groups:
        raise ValueError(
            f"{label}: key 'scale'/'units' has {len(scale)} entr(ies) but the regex "
            f"has {regex.groups} capture group(s)."
        )
    if scale is None and regex.groups > 2:
        raise ValueError(
            f"{label}: key regex has {regex.groups} capture groups; give a 'scale' "
            f"or 'units' to combine more than two."
        )
    out = np.empty(len(files), dtype=float)
    for i, name in enumerate(files):
        match = regex.search(Path(name).stem)
        if match is None:
            raise ValueError(f"{label}: key regex {pattern!r} did not match '{name}'.")
        groups = match.groups()
        if any(g is None for g in groups):
            raise ValueError(
                f"{label}: key regex {pattern!r} left an optional group unmatched "
                f"in '{name}'."
            )
        try:
            if scale is None:
                out[i] = float(".".join(groups))
            else:
                out[i] = sum(
                    int(g) * float(s) for g, s in zip(groups, scale, strict=True)
                )
        except ValueError as exc:
            raise ValueError(
                f"{label}: non-numeric key field in '{name}' (regex {pattern!r}): {exc}"
            ) from exc
    return out
