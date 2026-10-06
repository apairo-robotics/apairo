from __future__ import annotations

from pathlib import Path

import numpy as np

from apairo.core import AbstractDataset, AbstractLoader, FrameRef
from apairo.core.config import (
    CHANNELS_FILE,
    CONFIG_DIR,
    config_exists,
    declaration_exists,
    declaration_path,
    merge_declared_channels,
    read_config,
    read_declaration,
    safe_config_name,
    write_config,
)
from apairo.core.config import (
    register_raw_channel as _register_raw_channel,
)
from apairo.core.formats import Format, detect_format, find_format, formats, get_format
from apairo.core.naming import channel_frame_files
from apairo.core.sample import Sample
from apairo.dataset.registry import resolve_declaration
from apairo.loader import load_profile, load_timestamps, loads_timestamps
from apairo.utils.files import get_files
from apairo.utils.timestamps import get_end_of_time


def _detect_loader(channel_dir: Path) -> str | None:
    """The name of the format storing the channel in *channel_dir*, or ``None``
    -- asked of every registered format, in detection order."""
    fmt = detect_format(channel_dir)
    return fmt.name if fmt is not None else None


def _bare_channel_entries(directory: Path) -> dict[str, dict]:
    """Channels of a directory that holds its data files itself, with no
    sub-directory at all -- a channel directory opened on its own
    (``seq/velodyne_0``), or a folder of tables (a logger's CSV files).

    Frames of a directory format are one channel named after the directory,
    with their suffixed variants. Files of a format whose files are each a
    channel (a table) are one channel each, named after the file stem, or after
    the directory when the file is alone. Every entry reads from
    ``directory: "."``. Empty as soon as the directory has a sub-directory: that
    is a sequence or a root, and its channels are its sub-directories."""
    if any(p.is_dir() and not p.name.startswith(".") for p in directory.iterdir()):
        return {}
    name = directory.name
    entries: dict[str, dict] = {}
    frames = next(
        (f for f in formats() if not f.one_channel_per_file and f.detect(directory)),
        None,
    )
    if frames is not None:
        entries[name] = {"kind": "raw", "loader": frames.name, "directory": "."}
        for suffix, frag in _suffix_channel_entries(directory, frames.name).items():
            entries[f"{name}_{suffix}"] = {"kind": "raw", **frag, "directory": "."}
    per_file = [
        (fmt, p.name)
        for fmt in formats()
        if fmt.one_channel_per_file
        for p in fmt.data_files(directory)
    ]
    for fmt, filename in per_file:
        key = name if (len(per_file) == 1 and not entries) else Path(filename).stem
        if key in entries:
            key = filename.replace(".", "_")
        entries[key] = {
            "kind": "raw",
            "loader": fmt.name,
            "directory": ".",
            "array_file": filename,
        }
    return entries


def _declared_key_channels(directory: Path, *declares: str | Path | None) -> set[str]:
    """Channels whose stems a declaration explains with a ``key`` or ``order``
    regex -- their ``_`` is part of the name, not a suffix, so the scan must
    not fan them out into suffixed sub-channels. Union of the in-tree
    ``apairo.yaml`` and any external *declares* files."""
    declared: dict[str, dict] = {}
    if declaration_exists(directory):
        declared.update(read_declaration(declaration_path(directory)))
    for declare in declares:
        if declare is None:
            continue
        for k, v in read_declaration(declare).items():
            declared.setdefault(k, {}).update(v)
    return {k for k, v in declared.items() if v.get("key") or v.get("order")}


def _suffix_names(channel_dir: Path, fmt: Format) -> set[str]:
    """Suffixes of the variants beside a channel's frames: ``000000_intensity.npy``
    gives ``intensity``."""
    return {p.stem.split("_")[-1] for p in fmt.data_files(channel_dir) if "_" in p.stem}


def _suffix_channel_entries(channel_dir: Path, loader: str) -> dict[str, dict]:
    """Suffixed sub-channels found in *channel_dir*, keyed by suffix.

    A directory holding ``000000.npy`` *and* ``000000_intensity.npy`` yields
    ``{"intensity": {"loader": "npys", "directory": channel_dir.name, "suffix":
    "intensity"}}`` -- one sibling channel entry per suffix present, sharing
    *channel_dir* rather than owning a directory of its own. Empty (no fan-out)
    for a format without suffixed variants, or when none exist.
    """
    fmt = find_format(loader)
    if fmt is None or not fmt.suffixes:
        return {}
    return {
        suffix: {"loader": fmt.name, "directory": channel_dir.name, "suffix": suffix}
        for suffix in sorted(_suffix_names(channel_dir, fmt))
    }


class AsyncLayoutDataset(AbstractDataset):
    r"""Abstract *asynchronous layout* loader (one subdirectory per channel).

    This is the format primitive of the asynchronous dataset family. It is not
    a concrete dataset -- the synchronous KITTI-style datasets are
    :class:`~apairo.core.profiled_dataset.ProfiledDataset` subclasses (e.g.
    :class:`~apairo.dataset.semantic_kitti.SemanticKittiDataset`).

    It describes *how* channels are stored, never *which* channels exist: each
    channel is a subdirectory with its own ``timestamps.txt`` and data files in
    a registered format (``npys``, ``npy``, ``bin``, ``img``, ``zarr``, ``pcd``,
    ``csv``, or a plugin's -- see :mod:`apairo.core.formats`). A channel may instead carry its alignment key in its filenames --
    a ``key: {name: <regex>}`` / ``{file: <name>}`` spec parses it in memory at
    read time (nothing written), with an optional ``order`` enumeration policy;
    see ``docs/datasets/bring-your-own-dataset.md``. The set of channels is
    per-instance state, read from ``.apairo/channels.yaml`` (or an explicit
    ``dataset_profile``). Datasets
    with a *fixed* channel set layer a profile on top (e.g.
    :class:`~apairo.dataset.tartan_kitti.TartanKittiDataset`); datasets with
    *dynamic* channels (e.g. ``apairo-extractor`` output) use
    :class:`~apairo.dataset.raw.RawDataset`, which reads the channel set from
    ``.apairo`` with no profile.

    **Usage with an explicit profile (original API)**::

        ds = AsyncLayoutDataset(seq_dir, keys=["lidar", "cam"], dataset_profile="my.yaml")

    **Usage with** ``.apairo`` **(after** :meth:`init` **has been called)**::

        AsyncLayoutDataset.init(seq_dir)          # once, auto-detects channels
        ds = AsyncLayoutDataset(seq_dir)          # keys and loaders come from .apairo
        ds = AsyncLayoutDataset(seq_dir, keys=["lidar"])  # restrict to a subset

    Args:
        directory: Path to the dataset root / sequence directory.
        keys: Modality names to load.  ``None`` → all channels declared in
            ``.apairo`` (requires ``.apairo`` to exist).
        dataset_profile: YAML profile filename **or** absolute Path mapping keys
            to loader types.  ``None`` → loaders are read from ``.apairo``
            (requires ``.apairo`` to exist).
        declare: Path to a declaration file (the ``channels.yaml`` schema minus
            machine provenance) overlaid onto the channel metadata, per channel
            and per field.  Precedence: ``declare=`` > ``<directory>/apairo.yaml``
            > ``declare_base`` > ``.apairo/channels.yaml``.  Lets a read-only
            tree be declared from outside; apairo never writes a declaration.
        declare_base: Lower-precedence declaration overlaid *before* the
            in-tree ``apairo.yaml`` -- how a dataset root propagates its own
            ``<root>/apairo.yaml`` to every sequence (the sequence's own file
            stays the more specific word). Rarely passed by hand.
    """

    synchronous: bool = False

    def __init__(
        self,
        directory: str | Path,
        keys: list[str] | None = None,
        dataset_profile: str | Path | None = None,
        declare: str | Path | None = None,
        declare_base: str | Path | None = None,
    ) -> None:
        directory = Path(directory)
        declare = resolve_declaration(declare)  # a file, or a shipped name
        keys_defaulted = False  # True when keys=None resolved to "everything"

        # Channel metadata from .apairo (empty when a dataset_profile is passed
        # and no sidecar exists). alias_of maps an on-disk directory name to the
        # public name it is exposed under; timestamp_aliases maps a channel to the
        # one it borrows its clock from (its `timestamps_from`). Everything below
        # is keyed by the public name; the directory name only locates files.
        # _config_fallback is set by ConfigurableDataset when the directory is
        # read-only and the bootstrapped sidecar could not be written.
        fallback = getattr(self, "_config_fallback", None)
        if fallback is not None:
            channels = fallback.get("channels", {})
        elif config_exists(directory):
            channels = read_config(directory).get("channels", {})
        else:
            channels = {}
        # Human declarations overlay the machine registry, per channel and per
        # field, least specific first: the root's propagated declaration, the
        # sequence's own apairo.yaml, then an explicit declare= file.
        if declare_base is not None:
            channels = merge_declared_channels(channels, read_declaration(declare_base))
        if declaration_exists(directory):
            channels = merge_declared_channels(
                channels, read_declaration(declaration_path(directory))
            )
        if declare is not None:
            channels = merge_declared_channels(channels, read_declaration(declare))
        self._alias_of: dict[str, str] = {
            k: v["alias"] for k, v in channels.items() if v.get("alias")
        }
        # Honour the request's language: a channel explicitly asked for by its
        # real (directory) name is exposed under that name, one asked for by its
        # alias under the alias. Rewriting the alias table up front keeps every
        # table below in the request's vocabulary.
        if keys is not None:
            to_real = {alias: real for real, alias in self._alias_of.items()}
            for k in keys:
                real = to_real.get(k, k)
                if real in self._alias_of:
                    self._alias_of[real] = k
        self._timestamp_aliases: dict[str, str] = {
            self._public(k): self._resolve_key(v["timestamps_from"])
            for k, v in channels.items()
            if v.get("timestamps_from")
        }
        # A suffixed sub-channel (e.g. velodyne_0_intensity) has no directory of
        # its own -- it reads a suffix-filtered subset of another channel's files.
        self._suffix_of: dict[str, str] = {
            self._public(k): v["suffix"] for k, v in channels.items() if v.get("suffix")
        }
        # A channel may declare that its alignment key is parsed from its own
        # filenames (e.g. Rellis camera frame<N>-<epoch>_<ms>.jpg) instead of a
        # timestamps.txt -- the key is then computed in memory, nothing is written.
        self._key_spec: dict[str, dict] = {
            self._public(k): v["key"] for k, v in channels.items() if v.get("key")
        }
        # ...and how those files are enumerated/ordered, when the default frame-file
        # convention doesn't fit its naming. Separate from `key`; when absent it
        # defaults to the key's own regex.
        self._order_spec: dict[str, dict] = {
            self._public(k): v["order"] for k, v in channels.items() if v.get("order")
        }
        # A stacked `npy` channel may name the exact `.npy` it loads when its
        # directory colocates several (poses.npy beside valid_mask.npy) -- the
        # whole-array analogue of `suffix`. The name lives here, in the layout.
        self._array_file_of: dict[str, str] = {
            self._public(k): safe_config_name(
                v["array_file"], label=f"channel '{k}' array_file"
            )
            for k, v in channels.items()
            if v.get("array_file")
        }
        # A `pcd` channel's field contract. A PCD header is self-describing, so
        # the field set is a per-file property; declaring it here is what makes
        # the channel's width a layout decision instead of a per-file accident.
        self._fields_of: dict[str, list[str]] = {
            self._public(k): list(v["fields"])
            for k, v in channels.items()
            if v.get("fields")
        }
        # Each channel's whole entry, for its format to read what it needs --
        # including the fields a plugin format adds to the schema.
        self._meta_of: dict[str, dict] = {
            self._public(k): dict(v) for k, v in channels.items() if isinstance(v, dict)
        }

        if dataset_profile is not None:
            self._profile: dict[str, str] = load_profile(dataset_profile)
        elif channels:
            self._profile = {
                self._public(k): v["loader"]
                for k, v in channels.items()
                if "loader" in v
            }
            if keys is None:
                keys = sorted(self._profile.keys())
                keys_defaulted = True
        else:
            raise FileNotFoundError(
                f"No dataset_profile given and no .apairo or apairo.yaml found "
                f"in '{directory}'. Either pass dataset_profile=... or "
                f"declare=..., write an apairo.yaml declaration, or initialize "
                f"with {type(self).__name__}.init('{directory}')."
            )

        if keys is None:
            raise ValueError(
                "keys must be specified when dataset_profile is given. "
                "Pass keys=[...] or use .apairo (call init() first)."
            )

        # Re-key the on-disk directories by their public name so a request, a
        # loader and a sample all speak the same (aliased) language.
        self._files: dict[str, str] = {}
        for real, path in get_files(str(directory)).items():
            public = self._public(real)
            if public in self._files:
                raise ValueError(
                    f"Alias collision in '{directory}': the public name '{public}' "
                    f"is claimed by more than one channel. Clear one alias with "
                    f"`apairo alias <channel> --remove` (see `apairo status`)."
                )
            self._files[public] = path

        # A plain channel whose `directory` names where its files live, when
        # that is not the channel's own key: another top-level directory, or a
        # nested relative path (per_object_gt/pcd -- an annotation tool's
        # export). The explicit declaration wins over the same-name scan.
        for real, meta in channels.items():
            sub = meta.get("directory")
            if not sub or meta.get("suffix") or meta.get("array_file"):
                continue
            resolved = directory / safe_config_name(
                str(sub), label=f"channel '{real}' directory"
            )
            if resolved.is_dir():
                self._files[self._public(real)] = str(resolved)

        # Sub-channels that share another channel's directory instead of owning
        # one: a suffixed per-frame variant (*_intensity.npy), or a colocated
        # stacked array named by `array_file` (valid_mask.npy beside poses.npy).
        # Both read out of the directory named by their "directory" field.
        for real, meta in channels.items():
            if not (meta.get("suffix") or meta.get("array_file")):
                continue
            public = self._public(real)
            source_public = self._public(meta.get("directory", real))
            if source_public in self._files:
                self._files[public] = self._files[source_public]
            elif meta.get("directory"):
                # A directory that is not another channel's: the sequence itself
                # (`directory: "."` -- a table at the root, as TUM's
                # groundtruth.txt, or a bare channel directory opened on its own).
                resolved = directory / safe_config_name(
                    str(meta["directory"]), label=f"channel '{real}' directory"
                )
                if resolved.is_dir():
                    self._files[public] = str(resolved)

        keys = [self._resolve_key(k) for k in keys]
        if keys_defaulted:
            # Load-everything mode: a dataset-wide declaration (a root's file
            # is the union over its sequences) may name channels this sequence
            # does not hold -- skip them here instead of failing the sequence.
            # An explicitly requested channel still errors below.
            keys = [k for k in keys if k in self._files]
        missing = set(keys) - set(self._files)
        if missing:
            raise KeyError(f"Keys not found in dataset directory: {missing}")

        self._keys: list[str] = []
        self._set_keys(keys)
        self._init()

    # ------------------------------------------------------------------ alias

    def _public(self, real_name: str) -> str:
        """Public name a directory is exposed under (its alias, else itself)."""
        return self._alias_of.get(real_name, real_name)

    def _resolve_key(self, key: str) -> str:
        """Normalize a requested key (alias *or* real directory name) to its
        public name. Unknown keys pass through unchanged so the usual
        not-found error still fires."""
        if key in self._alias_of:  # a real name that has an alias -> its alias
            return self._alias_of[key]
        return key  # already a public name (an alias, or an unaliased real name)

    @classmethod
    def init(
        cls,
        directory: str | Path,
        *,
        raw_keys: list[str] | None = None,
        overwrite: bool = False,
        merge: bool = False,
        declare: str | Path | None = None,
    ) -> Path:
        """Scan an async-layout directory and write ``.apairo/channels.yaml``.

        All detected subdirectories are registered as raw channels.  Loader
        type is inferred from file extensions:

        * ``.bin`` → ``bin``
        * ``.pcd`` → ``pcd``
        * ``.png`` / ``.jpg`` / … → ``img``
        * multiple ``.npy`` files → ``npys``
        * single ``.npy`` file → ``npy``
        * a ``.csv`` table → ``csv`` (one row per frame)

        For ambiguous cases (e.g. a single-frame ``.npy`` that is actually
        per-frame), call :func:`~apairo.core.config.register_raw_channel`
        afterwards to override the detected loader.

        Args:
            directory: Dataset root / sequence directory to initialize.
            raw_keys: Subdirectory names to include.  ``None`` → all detected
                subdirectories with recognizable file types.
            overwrite: Discard the existing ``.apairo`` and rebuild from
                scratch.  Incompatible with ``merge``.
            merge: Add newly detected raw channels to an existing ``.apairo``
                without touching channels already declared (raw or
                preprocessed).  If ``.apairo`` does not yet exist, behaves
                like a normal init.  Incompatible with ``overwrite``.
            declare: Path to an external declaration file the scan should
                respect, in addition to the in-tree ``apairo.yaml`` (which is
                always read).  The scan only *reads* declarations -- it writes
                ``.apairo/`` and nothing else.

        Returns:
            Path of the written ``channels.yaml``.

        Raises:
            ValueError: If both ``overwrite`` and ``merge`` are ``True``.
            FileExistsError: If ``.apairo`` already exists and both
                ``overwrite`` and ``merge`` are ``False``.
            ValueError: If no new recognizable channels are found.
        """
        declare = resolve_declaration(declare)  # a file, or a shipped name
        if overwrite and merge:
            raise ValueError("overwrite and merge are mutually exclusive.")

        directory = Path(directory)

        explained = _declared_key_channels(directory, declare)

        if merge and config_exists(directory):
            existing = read_config(directory).get("channels", {})
            added = 0
            for channel_dir in sorted(directory.iterdir()):
                if not channel_dir.is_dir() or channel_dir.name.startswith("."):
                    continue
                if raw_keys is not None and channel_dir.name not in raw_keys:
                    continue
                loader = _detect_loader(channel_dir)
                if loader is None:
                    continue
                if channel_dir.name not in existing:
                    _register_raw_channel(directory, channel_dir.name, loader)
                    added += 1
                if channel_dir.name in explained:
                    continue
                # A directory's base channel may already be registered while a
                # suffix that only appeared later (e.g. *_intensity.npy) is not
                # -- check independently so re-running merge picks it up.
                for suffix, frag in _suffix_channel_entries(
                    channel_dir, loader
                ).items():
                    key = f"{channel_dir.name}_{suffix}"
                    if key in existing:
                        continue
                    _register_raw_channel(
                        directory,
                        key,
                        frag["loader"],
                        directory=frag["directory"],
                        suffix=frag["suffix"],
                    )
                    added += 1
            for key, entry in _bare_channel_entries(directory).items():
                if key in existing or (raw_keys is not None and key not in raw_keys):
                    continue
                _register_raw_channel(
                    directory,
                    key,
                    entry["loader"],
                    directory=entry["directory"],
                    suffix=entry.get("suffix"),
                    array_file=entry.get("array_file"),
                )
                added += 1
            if added == 0:
                detail = f" (checked: {raw_keys})" if raw_keys else ""
                raise ValueError(
                    f"No new recognizable channels found in '{directory}'{detail}."
                )
            return directory / CONFIG_DIR / CHANNELS_FILE

        if config_exists(directory) and not overwrite:
            raise FileExistsError(
                f".apairo already exists in '{directory}'. "
                f"Pass overwrite=True to reinitialize, or merge=True to add new channels."
            )

        channels: dict = {}
        for channel_dir in sorted(directory.iterdir()):
            if not channel_dir.is_dir() or channel_dir.name.startswith("."):
                continue
            if raw_keys is not None and channel_dir.name not in raw_keys:
                continue
            loader = _detect_loader(channel_dir)
            if loader is None:
                continue
            channels[channel_dir.name] = {
                "kind": "raw",
                "loader": loader,
            }
            if channel_dir.name in explained:
                continue
            for suffix, frag in _suffix_channel_entries(channel_dir, loader).items():
                channels[f"{channel_dir.name}_{suffix}"] = {"kind": "raw", **frag}
        # A directory holding its data files itself is a channel (or a set of
        # tables) of its own.
        for key, entry in _bare_channel_entries(directory).items():
            if raw_keys is None or key in raw_keys:
                channels[key] = entry

        if not channels:
            detail = f" (checked: {raw_keys})" if raw_keys else ""
            raise ValueError(
                f"No recognizable channels found in '{directory}'{detail}. "
                f"Expected subdirectories containing .bin, .pcd, .npy, .csv or "
                f"image files, or such files directly in the directory."
            )

        write_config(directory, {"version": 1, "channels": channels})
        return directory / CONFIG_DIR / CHANNELS_FILE

    # ------------------------------------------------------------------ keys

    @property
    def keys(self) -> list[str]:
        return self._keys

    @keys.setter
    def keys(self, keys: list[str]) -> None:
        keys = [self._resolve_key(k) for k in keys]
        missing = set(keys) - set(self._files)
        if missing:
            raise KeyError(f"Keys not found in dataset directory: {missing}")
        self._set_keys(list(keys))
        self._init()

    # ----------------------------------------------------------------- shape

    @property
    def shape(self) -> dict[str, tuple[int, ...]]:
        return {key: self.loaders[key].shape for key in self.keys}

    # ----------------------------------------------------------------- init

    def _init(self) -> None:
        if not self._keys:
            return
        self._init_loaders()
        self._init_timeline()

    def channel_format(self, key: str) -> str | None:
        """The name of the format channel *key* is stored in (its ``loader``),
        or ``None`` when the channel is unknown."""
        return self._profile.get(self._resolve_key(key))

    def _channel_meta(self, key: str) -> dict:
        """The channel's entry as the layout declares it -- what its format reads
        (``array_file``, ``fields``, ``key``, ``suffix``, plugin fields)."""
        meta = dict(getattr(self, "_meta_of", {}).get(key, {}))
        meta["loader"] = self._profile[key]
        if key in self._key_spec:
            meta["key"] = self._key_spec[key]
        if key in self._order_spec:
            meta["order"] = self._order_spec[key]
        if key in self._suffix_of:
            meta["suffix"] = self._suffix_of[key]
        if key in self._array_file_of:
            meta["array_file"] = self._array_file_of[key]
        if key in self._fields_of:
            meta["fields"] = self._fields_of[key]
        return meta

    def _init_loaders(self) -> None:
        loaders: dict[str, AbstractLoader] = {}
        for key in self._keys:
            fmt = get_format(self._profile[key])
            directory = self._files[key]
            meta = self._channel_meta(key)
            order_provider = getattr(self, "_order_providers", {}).get(key)
            if order_provider is not None:  # subclass callable: directory -> filenames
                files: list[str] | None = list(order_provider(directory))
            else:
                # An `order` / `key` regex or a suffix picks the files; otherwise
                # the format lists its own.
                files = channel_frame_files(
                    fmt, directory, meta, label=f"Channel '{key}'"
                )
            loaders[key] = fmt.open(Path(directory), meta, files)
        self.loaders: dict[str, AbstractLoader] = loaders
        self.timestamps: dict[str, np.ndarray] = self._collect_timestamps()
        self._check_clock_coverage()
        self.end_of_time: float = get_end_of_time(self.timestamps) + 1.0

    def _as_key_array(self, key: str, values) -> np.ndarray:
        """Validate + normalize a channel's key array: 1-D float, one value per
        frame, non-decreasing -- the timeline and ``synchronize()`` need each
        channel's keys in ascending order."""
        arr = np.atleast_1d(np.asarray(values, dtype=float)).ravel()
        n = len(self.loaders[key])
        if len(arr) != n:
            raise ValueError(
                f"Channel '{key}': its key provider returned {len(arr)} value(s) for "
                f"{n} frame(s)."
            )
        if arr.size > 1 and np.any(np.diff(arr) < 0):
            raise ValueError(
                f"Channel '{key}': keys are not non-decreasing. The timeline and "
                f"synchronize() need each channel's keys ascending -- check the "
                f"key/order regex captures the frame-ordering field."
            )
        return arr

    def _check_clock_coverage(self) -> None:
        """Every channel's clock must hold exactly one timestamp per frame.

        The timeline gives a channel one slot per timestamp, so a clock that does
        not match the frames is wrong in both directions: a short clock silently
        drops the trailing frames -- and a gap in the middle shifts every later
        frame onto the wrong timestamp -- while a long one only surfaces as an
        ``IndexError`` deep in ``_load``. Fail here, at construction, naming both
        counts. (A declarative ``key`` or a provider is checked where it is
        parsed, by ``_as_key_array``.)

        A suffixed sub-channel borrows the base channel's clock (shared
        directory): its ``*_<suffix>.npy`` files must cover every base frame."""
        for key in self._keys:
            suffix = self._suffix_of.get(key)
            if suffix is None:
                continue
            n_files = len(self.loaders[key])
            n_clock = len(self.timestamps[key])
            if n_files != n_clock:
                shared = Path(self._files[key]).name
                raise ValueError(
                    f"Suffixed sub-channel '{key}' has {n_files} '*_{suffix}.npy' "
                    f"file(s) in '{shared}/' but shares that channel's clock of "
                    f"{n_clock} frame(s): a suffixed variant must cover every base "
                    f"frame. Check for missing or extra '_{suffix}.npy' files."
                )
        # A colocated `array_file` sub-channel borrows its shared directory's
        # clock the same way a suffix channel does, but its stacked array's row
        # count is never validated -- a short array would only surface as a
        # cryptic IndexError deep in _load. Fail here at construction instead.
        for key in self._keys:
            array_file = self._array_file_of.get(key)
            if array_file is None:
                continue
            n_rows = len(self.loaders[key])
            n_clock = len(self.timestamps[key])
            if n_rows != n_clock:
                raise ValueError(
                    f"Colocated array_file sub-channel '{key}' has {n_rows} row(s) "
                    f"in '{array_file}' but shares a clock of {n_clock} frame(s): a "
                    f"colocated array must cover every frame."
                )
        # Every other channel: its own timestamps.txt, or the clock it borrows
        # with `timestamps_from`.
        for key in self._keys:
            if key in self._suffix_of or key in self._array_file_of:
                continue
            n_frames = len(self.loaders[key])
            n_clock = len(self.timestamps[key])
            if n_frames == n_clock:
                continue
            channel_dir = Path(self._files[key])
            src = self._timestamp_aliases.get(key)
            if src is not None and not (channel_dir / "timestamps.txt").exists():
                raise ValueError(
                    f"Channel '{key}' has {n_frames} frame(s) but borrows the clock "
                    f"of '{src}' (timestamps_from), which has {n_clock} "
                    f"timestamp(s): a borrowed clock must match the channel frame "
                    f"for frame. If '{key}' is not captured on '{src}'s ticks, give "
                    f"it its own clock (a timestamps.txt or a `key:`)."
                )
            raise ValueError(
                f"Channel '{key}' has {n_frames} frame(s) but {n_clock} timestamp(s) "
                f"in '{channel_dir.name}/timestamps.txt': every frame needs exactly "
                f"one. Look for a missing or extra data file, or a truncated "
                f"timestamps.txt."
            )

    def _collect_timestamps(self) -> dict[str, np.ndarray]:
        """Timestamps per loaded key: its own clock (a ``_key_providers`` callable,
        a declarative ``key`` spec, or a ``timestamps.txt``), else the clock of the
        channel named by its ``timestamps_from`` -- resolved through the *same*
        precedence, so borrowing works whatever the source's clock origin and
        regardless of the order channels are processed in."""
        timestamps: dict[str, np.ndarray] = {}
        fallback: list[str] = []

        def own_clock(key: str) -> np.ndarray | None:
            """A channel's own clock (provider > key spec > timestamps.txt), memoized
            in ``timestamps``; ``None`` if it has none of the three."""
            if key in timestamps:
                return timestamps[key]
            provider = getattr(self, "_key_providers", {}).get(key)
            if provider is not None:  # subclass callable: filenames -> key array
                timestamps[key] = self._as_key_array(
                    key, provider(getattr(self.loaders[key], "files", None))
                )
                return timestamps[key]
            if key in self._key_spec:  # declarative key, parsed in memory
                timestamps[key] = self._as_key_array(key, self._parse_key(key))
                return timestamps[key]
            ts_path = Path(self._files[key]) / "timestamps.txt"
            if ts_path.exists():
                timestamps[key] = load_timestamps(ts_path)
                return timestamps[key]
            return None

        for key in self._keys:
            if own_clock(key) is not None:
                continue
            if key in self._timestamp_aliases:  # timestamps_from a source channel
                src = self._timestamp_aliases[key]
                src_clock = own_clock(src)
                if src_clock is None:
                    raise ValueError(
                        f"'{key}' shares timestamps with '{src}' (timestamps_from), "
                        f"but '{src}' has no resolvable clock (no key spec, provider, "
                        f"or timestamps.txt)."
                    )
                timestamps[key] = src_clock
            else:
                fallback.append(key)
        if fallback:
            timestamps.update(loads_timestamps(fallback, self._files))
        return timestamps

    def _parse_key(self, key: str) -> np.ndarray:
        r"""A channel's alignment key from its ``key`` spec, computed in memory --
        nothing is written. The core's forms:

        - ``{name: '<regex>'}``: parse the key from each filename stem (a
          per-frame format). Capture groups become a number: with
          ``scale: [s0, s1, ...]`` as ``sum(int(group_i) * s_i)`` (e.g.
          ``<sec>_<ms>`` with ``scale [1, 0.001]``), else
          ``float('.'.join(groups))`` (one group = an index; two =
          ``<int>.<frac>``).
        - ``{file: '<name>'}``: read the keys from a named sidecar in the channel
          directory (one float per line -- a differently-named ``timestamps.txt``).

        Any other form is the format's own (a table's ``{column: ...}``), and
        its :meth:`~apairo.core.formats.Format.clock` computes it.
        """
        from apairo.core.keys import parse_filename_key

        spec = self._key_spec[key]
        label = f"Channel '{key}'"
        fmt = get_format(self._profile[key])
        own = [form for form in spec if form in fmt.key_forms]
        if own:
            return fmt.clock(self.loaders[key], spec, label)
        extra = [
            form for form in spec if form not in ("name", "file", "units", "scale")
        ]
        if extra:
            providers = [f.name for f in formats() if extra[0] in f.key_forms]
            raise ValueError(
                f"{label} declares a '{extra[0]}' key, which its loader "
                f"('{fmt.name}') does not provide"
                + (f" -- {', '.join(providers)} does." if providers else ".")
            )
        directory = Path(self._files[key])
        files = getattr(self.loaders[key], "files", None)
        if "file" not in spec and files is None:
            raise ValueError(
                f"{label} declares a filename-parsed key but its loader "
                f"('{fmt.name}') is stacked and has no per-frame filenames. "
                f"Filename keys need a per-frame loader."
            )
        return parse_filename_key(files or [], spec, directory=directory, label=label)

    def _init_timeline(self) -> None:
        """Build the interleaved timeline as two parallel numpy arrays."""
        from apairo.utils.timestamps import merge_timeline

        self._tl_key_idxs, self._tl_frame_idxs = merge_timeline(
            self.timestamps, self._keys
        )

    # ------------------------------------------------------------ dunder

    def __len__(self) -> int:
        return len(self._tl_key_idxs)

    def _load(self, idx: int) -> Sample:
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        key = self._keys[self._tl_key_idxs[idx]]
        frame = int(self._tl_frame_idxs[idx])
        return Sample(
            data={key: self.loaders[key][frame]},
            timestamp=float(self.timestamps[key][frame]),
        )

    # ------------------------------------------------------ frame provenance

    def _sequence_name(self) -> str | None:
        """This (single) sequence's directory name, or ``None`` if unknown."""
        d = getattr(self, "_sequence_dir", None)
        return d.name if d is not None else None

    def frame_info(self, idx: int) -> FrameRef:
        """Channel + row each interleaved event came from. See
        :meth:`AbstractDataset.frame_info`."""
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        return FrameRef(
            sequence=self._sequence_name(),
            channel=self._keys[self._tl_key_idxs[idx]],
            row=int(self._tl_frame_idxs[idx]),
        )

    @property
    def frame_sequence_ids(self) -> np.ndarray:
        """Sequence id per global event -- the sequence directory name (a single
        async dataset is one sequence). Object array of shape ``(len(self),)``."""
        return np.full(len(self), self._sequence_name(), dtype=object)

    @property
    def frame_channel_ids(self) -> np.ndarray:
        """Channel that produced each global event. Object array of shape
        ``(len(self),)``, vectorized from the merged timeline."""
        return np.asarray(self._keys, dtype=object)[self._tl_key_idxs]

    @property
    def frame_stems(self) -> np.ndarray:
        """Filename stem backing each global event: the per-frame data file's
        stem, or the zero-padded row for stacked (single-file) channels."""
        result = np.empty(len(self), dtype=object)
        for i in range(len(self)):
            key = self._keys[self._tl_key_idxs[i]]
            row = int(self._tl_frame_idxs[i])
            files = getattr(self.loaders[key], "files", None)
            result[i] = Path(files[row]).stem if files else f"{row:06d}"
        return result
