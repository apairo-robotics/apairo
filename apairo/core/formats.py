"""The format contract: how one storage format of a channel is read.

A :class:`Format` gathers everything the core needs to know about one way of
storing a channel -- which directories hold it, which files are its frames, the
loader that decodes one, the clock forms its data provides, the channel fields
it accepts, what ``apairo status`` shows and what ``apairo declare`` suggests.
The core asks the registry instead of naming formats, so a new format is a
plugin, not a core change:

.. code-block:: toml

    # the plugin's pyproject.toml
    [project.entry-points."apairo.formats"]
    xyz = "apairo_xyz:XyzFormat"

The built-in formats (``npy``, ``npys``, ``bin``, ``img``, ``zarr``, ``pcd``,
``csv``) implement the same contract, in :mod:`apairo.loader.formats`. See the
"Write a format plugin" guide.

What stays generic in the core, for every format: the filename and sidecar
``key`` forms of a per-frame format, ``timestamps.txt``, the clock checks,
``synchronize()`` and the views. A format only answers questions about its own
bytes.
"""

from __future__ import annotations

import importlib
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

if TYPE_CHECKING:
    from apairo.core.abstract_loader import AbstractLoader

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "apairo.formats"


@dataclass
class Facts:
    """What ``apairo status`` shows about a channel, read cheaply from disk.

    Any field left ``None`` prints as unknown. ``span`` -- the first and last
    timestamps -- is only set for a clock form the format itself provides (a
    table's column), and is all ``status`` needs of that clock: with
    ``frames``, it gives the rate. Filename and sidecar keys, and
    ``timestamps.txt``, are read by the core."""

    frames: int | None = None
    shape: list[int] | None = None
    dtype: str | None = None
    span: tuple[float, float] | None = None


class Format:
    """How one storage format of a channel is read.

    Subclass it, set the class attributes, override what differs from the
    defaults, and register an instance -- through the ``apairo.formats`` entry
    point group, or :func:`register_format` in process. The defaults describe
    a per-frame format: one file per frame, recognized by its extension, read
    by ``loader(directory)`` or ``loader(directory, files=...)``.

    Attributes:
        name: The loader name channels declare (``loader: <name>``).
        loader: The :class:`~apairo.core.abstract_loader.AbstractLoader` that
            decodes a frame. ``len(loader)`` is the frame count, ``loader[i]``
            frame ``i``; a per-frame loader also exposes ``files``, one
            filename per frame.
        extensions: Lower-case data-file suffixes, with the dot (``{".pcd"}``).
        per_frame: One file per frame. Only a per-frame format can take a
            filename ``key`` / ``order`` regex; a stacked one holds all frames
            in one object.
        one_channel_per_file: Each matching file is a channel of its own (a
            table), rather than all of them forming one channel.
        suffixes: Frames may carry suffixed variants (``000000_intensity.npy``)
            that fan out into sibling channels.
        fields: Optional channel fields the format reads (``array_file``,
            ``fields``, or its own).
        key_forms: Clock forms the data itself provides, beyond the core's
            filename and sidecar forms -- ``{"column"}`` for a table.
        priority: Detection order, lowest first, when several formats could
            claim a directory.
        write_suffix: The suffix of the frame files a per-frame format writes
            (``".png"`` for ``img``, which reads several).

    A format that can store a channel -- a preprocess output, a conversion --
    implements :meth:`write_frame` (per-frame) or :meth:`write_channel`
    (stacked), and :meth:`can_write`. Without them it is read-only. What it
    writes, it must read back unchanged.
    """

    name: ClassVar[str]
    loader: ClassVar[type[AbstractLoader]]
    extensions: ClassVar[frozenset[str]] = frozenset()
    per_frame: ClassVar[bool] = True
    one_channel_per_file: ClassVar[bool] = False
    suffixes: ClassVar[bool] = False
    fields: ClassVar[frozenset[str]] = frozenset()
    key_forms: ClassVar[frozenset[str]] = frozenset()
    priority: ClassVar[int] = 50
    write_suffix: ClassVar[str] = ""

    # ------------------------------------------------------------ discovery

    def data_files(self, directory: Path) -> list[Path]:
        """The files of *directory* this format reads, sorted by name -- never a
        dotfile or ``timestamps.txt``."""
        # Names are listed and sorted as strings: a channel can hold tens of
        # thousands of frames, and sorting Paths costs several times more.
        with os.scandir(directory) as entries:
            names = sorted(
                e.name
                for e in entries
                if not e.name.startswith(".")
                and e.name != "timestamps.txt"
                and e.is_file()
            )
        return [p for p in (directory / n for n in names) if self.matches(p)]

    def matches(self, path: Path) -> bool:
        """Is *path* a data file of this format?"""
        return path.suffix.lower() in self.extensions

    def detect(self, directory: Path) -> bool:
        """Does *directory* hold a channel of this format?"""
        return any(True for _ in self.data_files(directory))

    # ---------------------------------------------------------------- reading

    def open(
        self, directory: Path, meta: dict, files: list[str] | None = None
    ) -> AbstractLoader:
        """The loader for a channel stored in *directory*.

        *meta* is the channel's entry (``loader``, ``array_file``, ``fields``,
        ``key``, ...). *files* are the frame filenames the dataset resolved -- a
        ``key`` / ``order`` regex, a suffixed variant, a subclass provider --
        or ``None`` for the format's own listing."""
        if files is not None:
            return self.loader(str(directory), files=files)
        return self.loader(str(directory))

    def clock(self, loader: AbstractLoader, spec: dict, label: str) -> np.ndarray:
        """The clock for a ``key`` form in :attr:`key_forms`, one value per
        frame. Only called for those forms."""
        raise NotImplementedError(
            f"{label}: format '{self.name}' declares key forms "
            f"{sorted(self.key_forms)} but does not implement clock()."
        )

    # ---------------------------------------------------------------- writing

    @property
    def writes(self) -> bool:
        """Can this format store a channel? It does when it implements
        :meth:`write_frame` (per-frame) or :meth:`write_channel` (stacked)."""
        method = "write_frame" if self.per_frame else "write_channel"
        return getattr(type(self), method) is not getattr(Format, method)

    def can_write(self, array: np.ndarray) -> bool:
        """Can *array* -- one frame for a per-frame format, the whole channel
        ``(N, ...)`` for a stacked one -- be stored so that it reads back
        unchanged? Asked of the first output, before anything is written."""
        return self.writes

    def write_frame(self, path: Path, frame: np.ndarray) -> None:
        """Store one frame at *path*, ``<channel dir>/<stem><write_suffix>``."""
        raise NotImplementedError(f"format '{self.name}' does not write frames")

    def write_channel(self, directory: Path, name: str, array: np.ndarray) -> None:
        """Store channel *name* whole in *directory*, row ``i`` being frame
        ``i`` -- a file named after the channel, or the directory itself."""
        raise NotImplementedError(f"format '{self.name}' does not write channels")

    # ---------------------------------------------------- describing, checking

    def facts(
        self, directory: Path, meta: dict, files: list[str] | None = None
    ) -> Facts:
        """Frames, shape, dtype -- and the span of a clock form of its own --
        for ``apairo status``. By default, opened as loading opens it (*files*
        as in :meth:`open`), read off the first frame, and the span taken from
        :meth:`clock`. Override when that is not cheap. May raise: ``status``
        then shows the channel's facts as unknown."""
        loader = self.open(directory, meta, files)
        n = len(loader)
        if n == 0:
            return Facts(frames=0)
        first = np.asarray(loader[0])
        facts = Facts(frames=n, shape=list(first.shape), dtype=str(first.dtype))
        spec = meta.get("key")
        if isinstance(spec, dict) and set(spec) & self.key_forms:
            clock = np.asarray(self.clock(loader, spec, "status"), dtype=float)
            facts.span = (float(clock[0]), float(clock[-1]))
        return facts

    def declare_hints(self, directory: Path, meta: dict) -> list[str]:
        """Extra lines ``apairo declare`` writes under the channel (indented
        four spaces, commented unless confident). The filename ``key`` hint of
        a per-frame format is the core's."""
        return []

    def validate(self, key: str, meta: dict, storage_dir: Path) -> list[str]:
        """Issues with the format-specific fields *meta* holds, for ``check``.

        Check the fields that are there, never require one: ``check`` reads
        ``channels.yaml`` and the declaration apart, and a declaration may give
        a field the registry entry lacks. A field missing at load time is
        :meth:`open`'s to refuse, by name."""
        return []


# ── registry ────────────────────────────────────────────────────────────────

_FORMATS: dict[str, Format] = {}
_BUILTINS: frozenset[str] = frozenset()
_plugins_loaded = False


def register_format(fmt: Format | type[Format], *, replace: bool = False) -> Format:
    """Register a format under its :attr:`Format.name`. A name already taken is
    refused unless *replace* is set; a built-in format is never replaced."""
    if isinstance(fmt, type):
        fmt = fmt()
    name = getattr(fmt, "name", None)
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError(f"A format needs a plain `name`, got {name!r}.")
    if name in _FORMATS:
        if name in _BUILTINS:
            raise ValueError(f"'{name}' is a built-in format and cannot be replaced.")
        if not replace:
            raise ValueError(f"A format named '{name}' is already registered.")
    _FORMATS[name] = fmt
    return fmt


def _register_builtins() -> None:
    global _BUILTINS
    if _BUILTINS:
        return
    importlib.import_module("apairo.loader.formats")  # registers on import
    _BUILTINS = frozenset(_FORMATS)


def _load_plugins() -> None:
    """Register the formats installed through the ``apairo.formats`` entry
    point group, once. A plugin that fails to load is skipped with a warning:
    a broken plugin must not make every dataset unreadable."""
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True
    from importlib.metadata import entry_points

    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            fmt = ep.load()
            register_format(fmt)
        except Exception as exc:
            logger.warning(
                "apairo: format plugin '%s' was not loaded: %s", ep.name, exc
            )


def _ready() -> None:
    _register_builtins()
    _load_plugins()


def get_format(name: str) -> Format:
    """The registered format called *name*."""
    _ready()
    try:
        return _FORMATS[name]
    except KeyError:
        raise KeyError(
            f"Unknown loader '{name}'. Known formats: {', '.join(sorted(_FORMATS))}. "
            f"A format can be added as a plugin -- see the 'Write a format plugin' docs."
        ) from None


def find_format(name: Any) -> Format | None:
    """The registered format called *name*, or ``None``."""
    _ready()
    return _FORMATS.get(name) if isinstance(name, str) else None


def format_names() -> frozenset[str]:
    """Names of every registered format, built-in and plugin."""
    _ready()
    return frozenset(_FORMATS)


def formats() -> list[Format]:
    """Every registered format, in detection order."""
    _ready()
    return sorted(_FORMATS.values(), key=lambda f: (f.priority, f.name))


def detect_format(directory: Path) -> Format | None:
    """The format of the channel stored in *directory*, or ``None``. Formats
    whose files are each a channel of their own (tables) answer only when no
    other format does."""
    for fmt in formats():
        if fmt.detect(directory):
            return fmt
    return None
