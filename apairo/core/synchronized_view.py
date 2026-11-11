from __future__ import annotations

import functools
import math
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from apairo.core.abstract_dataset import AbstractDataset
from apairo.core.interpolator import Interpolator
from apairo.core.utils.indexing import from_end

if TYPE_CHECKING:
    from apairo.core.abstract_dataset import FrameRef
    from apairo.core.sample import Sample

# Matching strategy: (channel_timestamps, reference_timestamps) -> one event
# index per reference tick; a negative index marks "no match" (frame dropped).
SyncMethod = Callable[[np.ndarray, np.ndarray], np.ndarray]

# What a single channel can be synchronized with: a built-in matching mode,
# a custom matching callable, or a value-level interpolator.
# "latest" is a deprecated alias for "previous".
ChannelStrategy = (
    Literal["previous", "next", "nearest", "latest"] | SyncMethod | Interpolator
)

# How far, in seconds, a channel's event may lie from the tick: one bound for
# every channel, or one per channel (unlisted channels are unconstrained).
Tolerance = float | dict[str, float | None] | None


def _is_duration(value: Any) -> bool:
    """Whether *value* is a tolerance in seconds: a finite number ``>= 0``."""
    if isinstance(value, (bool, np.bool_)):
        return False
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return False
    return math.isfinite(value) and bool(value >= 0)


def _caller_depth() -> int:
    """Stack level of the first frame outside apairo, for a warning raised in
    :func:`warn_steps_not_synchronized` to point at the user's call."""
    import sys

    package = str(Path(__file__).resolve().parent.parent)
    depth = 1
    frame: Any = sys._getframe(1)
    while frame is not None and frame.f_code.co_filename.startswith(package):
        frame = frame.f_back
        depth += 1
    return depth


def warn_steps_not_synchronized(steps: list | None) -> None:
    """Warn that transform steps registered before ``synchronize()`` are not
    applied to the synchronized view, which reads the stored channels."""
    if steps:
        warnings.warn(
            f"synchronize() reads the stored channels of the dataset it is "
            f"called on: the {len(steps)} transform step(s) registered on it "
            f"before are not applied to the synchronized view. Register them "
            f"after: ds.synchronize(...).transform(...).",
            UserWarning,
            stacklevel=_caller_depth(),
        )


class SynchronizedView(AbstractDataset):
    """A synchronous view over an asynchronous dataset.

    Created by :meth:`~apairo.core.abstract_dataset.AbstractDataset.synchronize`.
    Index ``i`` returns a complete sample built around the *i*-th tick of the
    reference clock: every channel contributes either an existing event
    matched by timestamp, or a value synthesized at the tick by an
    :class:`~apairo.core.interpolator.Interpolator`.  The matching is a pure
    index computation (one ``np.searchsorted`` per channel at construction
    time) -- no data is read until access.

    Because the view is synchronous (``timestamps`` is ``None``), the full
    chaining API applies: ``.filter()``, ``.select()``, ``.cache()``,
    ``.join()``, and map-style PyTorch ``DataLoader`` with shuffling.

    .. note::
        The view reads channel data directly from the parent's loaders --
        transforms registered on the *parent* are not applied (they were
        written for single-event samples).  Register transforms on the view::

            ds_sync = ds.synchronize().transform("velodyne_0", RangeFilter(50))

    Args:
        parent: Asynchronous dataset exposing per-channel ``timestamps`` and
            ``loaders``.
        reference: The clock to resample onto.  Three forms:

            * channel name -- that channel's timestamps drive the view;
            * ``None`` -- the lowest-frequency channel is used;
            * **array of timestamps** -- an external clock.  Enables
              fixed-rate resampling (``np.arange(t0, t1, 1/hz)``) or
              distance-based resampling (see
              :func:`~apairo.utils.timestamps.clock_from_distance`).
        method: Strategy applied to every channel, or a dict mapping channel
            names to per-channel strategies (unlisted channels default to
            ``"previous"``).  A strategy is one of:

            * ``"previous"`` -- last event with ``t <= t_ref`` (zero-order
              hold, never looks into the future; ``"latest"`` is a
              deprecated alias);
            * ``"next"`` -- first event with ``t >= t_ref``;
            * ``"nearest"`` -- event closest in time to ``t_ref``, either
              side (ties favour the earlier event);
            * a **callable** ``(channel_ts, ref_ts) -> indices`` returning,
              for each reference tick, the event index to use (negative = no
              match, the frame is dropped);
            * an :class:`~apairo.core.interpolator.Interpolator` -- the value
              is synthesized at ``t_ref`` from the two bracketing events
              (continuous signals only: poses, IMU, commands).
        tolerance: Maximum ``|t - t_ref|`` in seconds, for every channel, or
            a dict mapping channel names to per-channel tolerances (unlisted
            channels, and channels mapped to ``None``, are unconstrained).
            For interpolated channels, *both* bracketing events must lie
            within the channel's tolerance. Reference ticks where any channel
            has no match are dropped.

    Example::

        from apairo_transform.interp import Se3Interp

        ds = TartanKittiDataset(seq, keys=["velodyne_0", "gicp_poses"])
        ds_sync = ds.synchronize(
            reference="velodyne_0",
            method={"gicp_poses": Se3Interp()},   # velodyne_0 -> "previous"
            tolerance={"gicp_poses": 0.15},       # velodyne_0: unconstrained
        )

        s = ds_sync[0]
        s.data["gicp_poses"]    # pose interpolated at s.timestamp
    """

    timestamps = None  # the view itself is synchronous

    def __init__(
        self,
        parent: AbstractDataset,
        reference: str | np.ndarray | None = None,
        method: ChannelStrategy | dict[str, ChannelStrategy] = "previous",
        tolerance: Tolerance = None,
    ) -> None:
        parent_ts = getattr(parent, "timestamps", None)
        name = parent.__class__.__name__
        if not isinstance(parent_ts, dict) or not parent_ts:
            if parent.is_synchronous:
                raise ValueError(
                    f"synchronize() requires an asynchronous dataset with "
                    f"per-channel timestamps; this {name} is already synchronous."
                )
            raise ValueError(
                f"synchronize() needs each channel's clock, and this {name} "
                f"(a view over an asynchronous dataset) does not carry them. "
                f"Call synchronize() on the dataset itself, then filter or "
                f"select the result."
            )
        if not isinstance(getattr(parent, "loaders", None), dict):
            raise ValueError(
                f"synchronize() reads each channel's frames by row, and this "
                f"{name} does not expose them. Synchronize each dataset before "
                f"combining them: a.synchronize(...).concat(b.synchronize(...))."
            )
        warn_steps_not_synchronized(getattr(parent, "_pipeline", None))

        keys = list(parent.keys)
        strategies = self._resolve_strategies(method, keys)
        tolerances = self._resolve_tolerances(tolerance, keys)
        ref_name, ref_ts = self._resolve_clock(reference, parent_ts, keys)

        valid = np.ones(len(ref_ts), dtype=bool)
        index_map: dict[str, np.ndarray] = {}
        channel_ts: dict[str, np.ndarray] = {}

        for key in keys:
            ts = np.asarray(parent_ts[key], dtype=float)
            channel_ts[key] = ts
            idx, ok = self._match(strategies[key], ts, ref_ts, tolerances[key])
            valid &= ok
            index_map[key] = idx

        keep = np.where(valid)[0]
        self._parent = parent
        self._reference = ref_name
        self._method = method
        self._strategies = strategies
        self._tolerance = dict(tolerance) if isinstance(tolerance, dict) else tolerance
        self._ref_timestamps = ref_ts[keep]
        self._index_map = {k: v[keep].astype(np.intp) for k, v in index_map.items()}
        self._channel_ts = channel_ts
        self._keys = keys

    @classmethod
    def _from_state(
        cls,
        parent: AbstractDataset,
        *,
        reference: str | None,
        strategies: dict[str, Any],
        tolerance: Tolerance,
        reference_timestamps: np.ndarray,
        index_map: dict[str, np.ndarray],
    ) -> SynchronizedView:
        """A view rebuilt from its persisted state -- the matching is not
        recomputed (see :func:`apairo.core.views.load_view`)."""
        view = cls.__new__(cls)
        view._parent = parent
        view._reference = reference
        view._method = dict(strategies)
        view._strategies = dict(strategies)
        view._tolerance = dict(tolerance) if isinstance(tolerance, dict) else tolerance
        view._ref_timestamps = np.asarray(reference_timestamps, dtype=float)
        view._index_map = {
            k: np.asarray(v, dtype=np.intp) for k, v in index_map.items()
        }
        view._channel_ts = {
            k: np.asarray(parent.timestamps[k], dtype=float)  # type: ignore[index]
            for k in index_map
        }
        view._keys = list(index_map)
        return view

    def persist(self, name: str, *, overwrite: bool = False) -> Path:
        """Freeze this view in its sequence's ``.apairo`` under *name*: the
        reference clock, one index array per channel, the method, the
        tolerance, and a fingerprint of every source channel. No data is
        copied. Reload it -- the same frames, without recomputing the matching
        -- with ``RawDataset(seq).load_view(name)``; a source changed since is
        refused, by name.

        Raises:
            FileExistsError: A view of that name exists and *overwrite* is
                not set.
            TypeError: The view was not synchronized directly on a sequence.
        """
        from apairo.core.views import persist_view

        return persist_view(self, name, overwrite=overwrite)

    # ------------------------------------------------------------- resolution

    @staticmethod
    def _resolve_strategies(method, keys: list[str]) -> dict[str, ChannelStrategy]:
        """Normalize *method* into one validated strategy per channel."""
        if isinstance(method, (set, frozenset)):
            raise TypeError(
                f"method must be a single strategy or a dict {{channel: strategy}}, "
                f"not a set -- did you write {{a, b}} instead of {{a: b}}? "
                f"Got {method!r}."
            )
        if isinstance(method, dict):
            unknown = set(method) - set(keys)
            if unknown:
                raise KeyError(
                    f"method maps unknown channels {sorted(unknown)}; "
                    f"dataset keys are {keys}."
                )
            strategies = {k: method.get(k, "previous") for k in keys}
        else:
            strategies = {k: method for k in keys}

        if any(s == "latest" for s in strategies.values()):
            warnings.warn(
                "method='latest' is deprecated, use 'previous' (same "
                "semantics: last event with t <= t_ref).",
                DeprecationWarning,
                stacklevel=4,  # _resolve_strategies <- __init__ <- synchronize()
            )
            strategies = {
                k: ("previous" if s == "latest" else s) for k, s in strategies.items()
            }

        for key, strat in strategies.items():
            if isinstance(strat, Interpolator) or callable(strat):
                continue
            if strat not in ("previous", "next", "nearest"):
                raise ValueError(
                    f"Strategy for {key!r} must be 'previous', 'next', "
                    f"'nearest', a callable (channel_ts, ref_ts) -> indices, "
                    f"or an Interpolator, got {strat!r}"
                )
        return strategies

    @staticmethod
    def _resolve_tolerances(tolerance, keys: list[str]) -> dict[str, float | None]:
        """Normalize *tolerance* into one validated bound (or ``None``) per
        channel, as :meth:`_resolve_strategies` does for *method*."""
        if isinstance(tolerance, (set, frozenset)):
            raise TypeError(
                f"tolerance must be a number of seconds or a dict "
                f"{{channel: seconds}}, not a set -- did you write {{a, b}} "
                f"instead of {{a: b}}? Got {tolerance!r}."
            )
        if isinstance(tolerance, dict):
            unknown = set(tolerance) - set(keys)
            if unknown:
                raise KeyError(
                    f"tolerance maps unknown channels {sorted(unknown)}; "
                    f"dataset keys are {keys}."
                )
            given = {k: tolerance.get(k) for k in keys}
        else:
            given = {k: tolerance for k in keys}
        for key, value in given.items():
            if value is None or _is_duration(value):
                continue
            where = f" for {key!r}" if isinstance(tolerance, dict) else ""
            raise ValueError(
                f"The tolerance{where} must be a finite number of seconds >= 0, "
                f"or None for no bound; got {value!r}."
            )
        return {k: None if v is None else float(v) for k, v in given.items()}

    @staticmethod
    def _resolve_clock(reference, parent_ts: dict, keys: list[str]):
        """Resolve *reference* into ``(name_or_None, timestamp_array)``.

        Accepts a channel name, ``None`` (lowest-frequency channel), or an
        explicit array of timestamps — an external clock (e.g. fixed-rate
        ticks or distance-based ticks from odometry).
        """
        if reference is None:
            from apairo.utils.timestamps import get_reference_timestamps

            name = get_reference_timestamps({k: parent_ts[k] for k in keys})
            return name, np.asarray(parent_ts[name], dtype=float)

        if isinstance(reference, str):
            if reference not in keys:
                raise KeyError(
                    f"Reference channel {reference!r} not in dataset keys {keys}."
                )
            return reference, np.asarray(parent_ts[reference], dtype=float)

        ref_ts = np.asarray(reference, dtype=float)
        if ref_ts.ndim != 1 or len(ref_ts) == 0:
            raise ValueError(
                f"An external clock must be a non-empty 1-D array of "
                f"timestamps, got shape {ref_ts.shape}."
            )
        if np.any(np.diff(ref_ts) < 0):
            raise ValueError("External clock timestamps must be ascending.")
        return None, ref_ts

    # --------------------------------------------------------------- matching

    @staticmethod
    def _match(
        strat: ChannelStrategy,
        ts: np.ndarray,
        ref_ts: np.ndarray,
        tolerance: float | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute event indices and per-tick validity for one channel.

        Returns ``(idx, valid)`` where ``idx`` has shape ``(N,)`` for matching
        strategies or ``(N, 2)`` (bracketing pair) for interpolators.
        """
        right = np.searchsorted(ts, ref_ts, side="right")
        latest = right - 1  # last event with t <= t_ref; -1 when none yet

        if isinstance(strat, Interpolator):
            i0 = np.clip(latest, 0, len(ts) - 1)
            i1 = np.clip(right, 0, len(ts) - 1)
            # exact matches collapse to a single index -- the stored value is
            # returned directly, the interpolator is never called
            i1 = np.where(ts[i0] == ref_ts, i0, i1)
            # bracketed: an event at or before the tick, and one at or after
            valid = (latest >= 0) & (ts[i1] >= ref_ts)
            if tolerance is not None:
                valid &= np.maximum(ref_ts - ts[i0], ts[i1] - ref_ts) <= tolerance
            return np.stack([i0, i1], axis=1), valid

        if callable(strat):
            idx = np.asarray(strat(ts, ref_ts))
            if idx.shape != ref_ts.shape:
                raise ValueError(
                    f"Custom method returned shape {idx.shape}, expected "
                    f"{ref_ts.shape} (one index per reference tick; "
                    f"negative = no match)."
                )
            valid = (idx >= 0) & (idx < len(ts))
        elif strat == "previous":
            idx = latest
            valid = latest >= 0
        elif strat == "next":
            # first event with t >= t_ref; len(ts) when none remains
            idx = np.searchsorted(ts, ref_ts, side="left")
            valid = idx < len(ts)
        else:  # nearest -- either side; ties favour the earlier event
            prev = np.clip(latest, 0, len(ts) - 1)
            nxt = np.clip(right, 0, len(ts) - 1)
            idx = np.where(
                np.abs(ts[prev] - ref_ts) <= np.abs(ts[nxt] - ref_ts),
                prev,
                nxt,
            )
            valid = np.ones(len(ref_ts), dtype=bool)

        idx = np.clip(idx, 0, len(ts) - 1)
        if tolerance is not None:
            valid = valid & (np.abs(ts[idx] - ref_ts) <= tolerance)
        return idx, valid

    # ------------------------------------------------------------- properties

    @property
    def root_dir(self) -> Path | None:
        """Delegates to the parent's ``root_dir`` (or ``None`` if it has
        none), so ``.calibration`` still resolves on a synchronized view --
        without this, every synchronized view read an empty ``Calibration``
        regardless of what its parent exposed."""
        return getattr(self._parent, "root_dir", None)

    @property
    def reference(self) -> str | None:
        """Channel providing the clock, or ``None`` for an external clock."""
        return self._reference

    @property
    def reference_timestamps(self) -> np.ndarray:
        """Timestamp of each frame in the view (reference clock)."""
        return self._ref_timestamps

    @property
    def frame_indices(self) -> dict[str, np.ndarray]:
        """Per-channel event indices backing each frame.

        Shape ``(n,)`` for matched channels (the event used), ``(n, 2)`` for
        interpolated channels (the bracketing pair).
        """
        return self._index_map

    def time_offsets(self, key: str) -> np.ndarray:
        """Signed ``t_event - t_ref`` per frame for *key*, in seconds.

        Interpolated channels return zeros: their values are synthesized at
        the reference instant.
        """
        if isinstance(self._strategies[key], Interpolator):
            return np.zeros(len(self), dtype=float)
        ts = self._channel_ts[key]
        return ts[self._index_map[key]] - self._ref_timestamps

    def frame_info(self, idx: int) -> FrameRef:
        """Provenance of synchronised frame *idx*.

        A synchronised frame is *composite*: each channel is backed by its own
        source event (see :attr:`frame_indices`), so there is no single origin
        channel -- ``channel`` is ``None`` and ``row`` is the **view index**, not
        an on-disk row (unlike a profiled dataset, where ``row`` addresses a file
        on disk). ``sequence`` is filled when the parent belongs to a single
        sequence (see :attr:`frame_sequence_ids`), ``None`` otherwise.

        For real per-frame provenance use :attr:`frame_indices`;
        ``frame_indices[self.reference][idx]`` is the reference-clock event this
        tick was resampled onto.
        """
        return super().frame_info(idx)

    @functools.cached_property
    def frame_sequence_ids(self) -> np.ndarray:
        """Sequence id per synchronised frame, shape ``(n,)``.

        A synchronised frame mixes events from several channels, so it only
        carries a sequence id when the whole parent belongs to one sequence;
        over a multi-sequence parent this raises ``AttributeError``, keeping
        the availability probe of the base class."""
        parent_ids = self._parent.frame_sequence_ids
        if len(parent_ids) and (parent_ids != parent_ids[0]).any():
            raise AttributeError(
                f"{type(self).__name__} exposes frame_sequence_ids only over "
                f"a single-sequence parent; this parent spans "
                f"{len(set(parent_ids))} sequences."
            )
        seq = parent_ids[0] if len(parent_ids) else None
        return np.full(len(self), seq, dtype=object)

    @property
    def frame_channel_ids(self) -> np.ndarray:
        """Unavailable: a synchronised frame composites every channel (see
        :meth:`frame_info`), so there is no single origin channel -- unlike
        :attr:`frame_sequence_ids`, this raises unconditionally, even over a
        single-sequence parent. Use :attr:`frame_indices` for real per-channel
        provenance."""
        raise AttributeError(
            f"{type(self).__name__} exposes no frame_channel_ids: a "
            "synchronised frame composites every channel, so there is no "
            "single origin. Use frame_indices for per-channel provenance."
        )

    # ----------------------------------------------------------------- access

    def __len__(self) -> int:
        return len(self._ref_timestamps)

    def _load(self, idx: int) -> Sample:
        from apairo.core.sample import Sample

        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        data = {key: self._channel_value(key, idx) for key in self._keys}
        return Sample(data=data, timestamp=float(self._ref_timestamps[idx]))

    def load(self, key: str, idx: int):
        """Channel *key* of frame *idx*, read alone: the event matched to the
        frame's tick, or the value interpolated at it. Only that channel's
        events are read -- the cheap way to look at one channel of a frame
        (``self[idx]`` reads every channel). Like ``load`` on any dataset, the
        stored value: transforms registered on the view are not applied.

        Raises:
            KeyError: *key* is not one of the view's channels.
            IndexError: *idx* is out of range.
        """
        if key not in self._index_map:
            raise KeyError(
                f"{key!r} is not a channel of this view; its channels are {self._keys}."
            )
        if isinstance(idx, (int, np.integer)) and idx < 0:
            idx = from_end(int(idx), len(self))
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        return self._channel_value(key, int(idx))

    def _reads_channel(self, key: str) -> bool:
        return key in self._index_map

    def _channel_value(self, key: str, idx: int):
        """Frame *idx*'s value for *key*, reading that channel only."""
        loader = self._parent.loaders[key]
        strat = self._strategies[key]
        if not isinstance(strat, Interpolator):
            return loader[int(self._index_map[key][idx])]
        i0, i1 = (int(i) for i in self._index_map[key][idx])
        v0 = loader[i0]
        if i0 == i1:  # exact match -- no synthesis needed
            return v0
        ts = self._channel_ts[key]
        t_ref = float(self._ref_timestamps[idx])
        return strat(t_ref, float(ts[i0]), v0, float(ts[i1]), loader[i1])

    def __repr__(self) -> str:
        ref = self._reference if self._reference is not None else "<external clock>"
        method: object
        if isinstance(self._method, dict):
            method = "per-channel"
        else:
            m: Any = self._method
            method = getattr(m, "__name__", m)
        return (
            f"SynchronizedView(n={len(self)}, reference={ref!r}, "
            f"method={method!r}, keys={self._keys})"
        )
