from __future__ import annotations

from pathlib import Path

import numpy as np

from apairo.core import AbstractDataset, AbstractLoader, FrameRef
from apairo.core.config import (
    CHANNELS_FILE,
    CONFIG_DIR,
    config_exists,
    read_config,
    safe_config_name,
    write_config,
)
from apairo.core.config import (
    register_raw_channel as _register_raw_channel,
)
from apairo.core.naming import suffixed_frame_files
from apairo.core.sample import Sample
from apairo.loader import load_profile, load_timestamps, loads_timestamps, str_to_loader
from apairo.utils.files import get_files
from apairo.utils.timestamps import get_end_of_time


def _detect_loader(channel_dir: Path) -> str | None:
    """Infer loader type from the contents of *channel_dir*.

    A channel directory that is itself a Zarr array store (it holds a
    ``.zarray`` / ``zarr.json`` metadata file, with ``timestamps.txt`` placed
    beside the chunks) is detected as ``"zarr"``; otherwise the loader is
    inferred from the data-file extensions.
    """
    if (channel_dir / ".zarray").exists() or (channel_dir / "zarr.json").exists():
        return "zarr"
    data_files = [
        f for f in channel_dir.iterdir() if f.is_file() and f.name != "timestamps.txt"
    ]
    if not data_files:
        return None
    exts = {f.suffix.lower() for f in data_files}
    if ".bin" in exts:
        return "bin"
    if exts & {".png", ".jpg", ".jpeg", ".bmp"}:
        return "img"
    npy_files = [f for f in data_files if f.suffix == ".npy"]
    if npy_files:
        # Multiple per-frame files → npys; single file → npy.
        return "npys" if len(npy_files) > 1 else "npy"
    return None


def _suffix_channel_entries(channel_dir: Path, loader: str) -> dict[str, dict]:
    """Suffixed npy sub-channels found in *channel_dir*, keyed by suffix.

    A directory holding ``000000.npy`` *and* ``000000_intensity.npy`` yields
    ``{"intensity": {"loader": "npys", "directory": channel_dir.name, "suffix":
    "intensity"}}`` -- one sibling channel entry per suffix present, sharing
    *channel_dir* rather than owning a directory of its own. Empty (no fan-out)
    for any loader other than ``"npys"``, or when no suffixed files exist.
    """
    if loader != "npys":
        return {}
    from apairo.utils import npy_analyser

    return {
        suffix: {"loader": "npys", "directory": channel_dir.name, "suffix": suffix}
        for suffix in sorted(npy_analyser(channel_dir) - {""})
    }


class AsyncLayoutDataset(AbstractDataset):
    r"""Abstract *asynchronous layout* loader (one subdirectory per channel).

    This is the format primitive of the asynchronous dataset family. It is not
    a concrete dataset -- the synchronous KITTI-style datasets are
    :class:`~apairo.core.profiled_dataset.ProfiledDataset` subclasses (e.g.
    :class:`~apairo.dataset.semantic_kitti.SemanticKittiDataset`).

    It describes *how* channels are stored, never *which* channels exist: each
    channel is a subdirectory with its own ``timestamps.txt`` and data files in
    a format known to the loader registry (``npys``, ``npy``, ``bin``, ``img``,
    ``zarr``). A channel may instead carry its alignment key in its filenames --
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
    """

    synchronous: bool = False

    def __init__(
        self,
        directory: str | Path,
        keys: list[str] | None = None,
        dataset_profile: str | Path | None = None,
    ) -> None:
        directory = Path(directory)

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
        else:
            raise FileNotFoundError(
                f"No dataset_profile given and no .apairo found in '{directory}'. "
                f"Either pass dataset_profile=..., or initialize with "
                f"{type(self).__name__}.init('{directory}')."
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

        keys = [self._resolve_key(k) for k in keys]
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
    ) -> Path:
        """Scan an async-layout directory and write ``.apairo/channels.yaml``.

        All detected subdirectories are registered as raw channels.  Loader
        type is inferred from file extensions:

        * ``.bin`` → ``bin``
        * ``.png`` / ``.jpg`` / … → ``img``
        * multiple ``.npy`` files → ``npys``
        * single ``.npy`` file → ``npy``

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

        Returns:
            Path of the written ``channels.yaml``.

        Raises:
            ValueError: If both ``overwrite`` and ``merge`` are ``True``.
            FileExistsError: If ``.apairo`` already exists and both
                ``overwrite`` and ``merge`` are ``False``.
            ValueError: If no new recognizable channels are found.
        """
        if overwrite and merge:
            raise ValueError("overwrite and merge are mutually exclusive.")

        directory = Path(directory)

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
            for suffix, frag in _suffix_channel_entries(channel_dir, loader).items():
                channels[f"{channel_dir.name}_{suffix}"] = {"kind": "raw", **frag}

        if not channels:
            detail = f" (checked: {raw_keys})" if raw_keys else ""
            raise ValueError(
                f"No recognizable channels found in '{directory}'{detail}. "
                f"Expected subdirectories containing .bin, .npy, or image files."
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

    def _init_loaders(self) -> None:
        loaders: dict[str, AbstractLoader] = {}
        for key in self._keys:
            loader_cls = str_to_loader[self._profile[key]]
            directory = self._files[key]
            suffix = self._suffix_of.get(key)
            order_provider = getattr(self, "_order_providers", {}).get(key)
            enumerate_by_regex = key in self._order_spec or (
                key in self._key_spec and "name" in self._key_spec[key]
            )
            if order_provider is not None:  # subclass callable: directory -> filenames
                loaders[key] = loader_cls(
                    directory, files=list(order_provider(directory))
                )
            elif enumerate_by_regex:
                # Declarative enumeration policy (the `order` regex, else the `key`
                # regex): a channel whose names carry a '_' (a Rellis <epoch>_<ms>),
                # which the default frame-file convention reserves for suffixes, still
                # enumerates, and the loader's own name sort is bypassed.
                if self._profile[key] not in {"npys", "img", "bin"}:
                    raise ValueError(
                        f"Channel '{key}' declares a filename key/order but its loader "
                        f"'{self._profile[key]}' has no per-frame files -- filename "
                        f"keys/order need a per-frame loader (npys, img, bin)."
                    )
                loaders[key] = loader_cls(
                    directory, files=self._enumerate(key, directory)
                )
            elif suffix:
                loaders[key] = loader_cls(
                    directory, files=suffixed_frame_files(directory, suffix)
                )
            elif self._array_file_of.get(key) and self._profile[key] == "npy":
                # A colocated stacked array named explicitly (valid_mask.npy in a
                # shared gicp_poses/): load that file, not the directory's glob[0].
                loaders[key] = loader_cls(directory, file=self._array_file_of[key])
            else:
                loaders[key] = loader_cls(directory)
        self.loaders: dict[str, AbstractLoader] = loaders
        self.timestamps: dict[str, np.ndarray] = self._collect_timestamps()
        self._check_suffix_coverage()
        self.end_of_time: float = get_end_of_time(self.timestamps) + 1.0

    def _enumerate(self, key: str, directory: str) -> list[str]:
        """Ordered filenames for a channel with a declarative enumeration policy:
        the loader-extension files whose stem matches its ``order`` regex (else its
        ``key`` regex), sorted by the numeric value of the regex's first capture
        group (else lexicographically). This is the ``order`` contract -- it lets a
        channel whose names carry a '_' (a Rellis ``<epoch>_<ms>``, which the default
        frame-file convention reserves for suffixes) enumerate anyway, filters out
        strays (a ``timestamps.txt``, a dotfile, a wrong-extension note), and orders
        even non-zero-padded frame indices correctly."""
        import re

        spec = self._order_spec.get(key) or self._key_spec.get(key, {})
        pattern = spec.get("name")
        if pattern is None:
            raise ValueError(
                f"Channel '{key}' needs an 'order' or 'key' regex ('name') to "
                f"enumerate by; got {spec!r}."
            )
        regex = re.compile(pattern)
        exts = {
            "npys": {".npy"},
            "npy": {".npy"},
            "bin": {".bin"},
            "img": {".png", ".jpg", ".jpeg", ".bmp"},
        }.get(self._profile[key])

        def matched(p: Path) -> bool:
            if not p.is_file() or p.name == "timestamps.txt" or p.name.startswith("."):
                return False
            if exts is not None and p.suffix.lower() not in exts:
                return False
            return regex.search(p.stem) is not None

        def order_key(name: str) -> tuple[int, str]:
            match = regex.search(Path(name).stem)
            first = match.groups()[0] if (match and match.groups()) else None
            return (int(first) if (first and first.isdigit()) else 0, name)

        names = sorted(
            (p.name for p in Path(directory).iterdir() if matched(p)), key=order_key
        )
        if not names:
            raise FileNotFoundError(
                f"Channel '{key}': no files in '{directory}' match the enumeration "
                f"regex {pattern!r}."
            )
        return names

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

    def _check_suffix_coverage(self) -> None:
        """A suffixed sub-channel borrows the base channel's clock (shared
        directory), so the timeline gives it one slot per base frame. If its
        ``*_<suffix>.npy`` files don't cover every base frame, ``_load`` would
        index past the loader -- fail here, at construction, with a clear message
        instead of a cryptic ``IndexError`` later."""
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
        nothing is written. Two forms:

        - ``{name: '<regex>'}``: parse the key from each filename stem. Capture
          groups become a number: with ``scale: [s0, s1, ...]`` as
          ``sum(int(group_i) * s_i)`` (e.g. ``<sec>_<ms>`` with ``scale [1, 0.001]``),
          else ``float('.'.join(groups))`` (one group = an index; two = ``<int>.<frac>``).
        - ``{file: '<name>'}``: read the keys from a named sidecar in the channel
          directory (one float per line -- a differently-named ``timestamps.txt``).
        """
        from apairo.core.keys import parse_filename_key

        spec = self._key_spec[key]
        directory = Path(self._files[key])
        files = getattr(self.loaders[key], "files", None)
        if "file" not in spec and files is None:
            raise ValueError(
                f"Channel '{key}' declares a filename-parsed key but its loader "
                f"('{self._profile[key]}') is stacked and has no per-frame filenames. "
                f"Filename keys need a per-frame loader (npys/img/bin)."
            )
        return parse_filename_key(
            files or [], spec, directory=directory, label=f"Channel '{key}'"
        )

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
