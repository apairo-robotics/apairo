"""RootSequenceMixin -- shared *dataset root* behaviour for the async family.

Several asynchronous datasets accept either a single sequence directory or a
*root* directory holding many sequences (``TartanKittiDataset``,
``RawDataset``).  The root behaviour is identical between them -- flat indexing
over the concatenated sequences, per-sequence access, and per-sequence
synchronization -- so it lives here once instead of being copied per dataset.

A subclass decides *single sequence vs root* in its ``__init__``.  For the root
case it calls :meth:`_init_root` with a factory that builds one single-sequence
instance of the same class; everything else (``__len__``, ``_load``,
``synchronize``, ``keys`` propagation, ``sequences``/``sequence()``) is provided
here and dispatches on ``self._is_root``, delegating the single-sequence path to
``super()`` (the layout base, e.g.
:class:`~apairo.dataset.async_layout.AsyncLayoutDataset`).

The mixin must precede the layout base in the MRO::

    class RawDataset(RootSequenceMixin, AsyncLayoutDataset, ConfigurableDataset):
        ...

Subclass contract:

* single-sequence instances expose ``self._sequence_dir`` (a ``Path``);
* subclasses implement :meth:`_single_available` (channels of one sequence) and
  :meth:`_set_single_keys` (apply keys to one sequence).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from apairo.core.abstract_dataset import FrameRef
from apairo.core.config import Calibration, read_calibration

if TYPE_CHECKING:
    # For the type checker the mixin sits on AbstractDataset (its real MRO
    # position); at runtime it stays a plain mixin so the MRO is unchanged.
    from apairo.core.abstract_dataset import AbstractDataset as _MixinBase
    from apairo.core.sequence_view import SequenceView
else:
    _MixinBase = object


class RootSequenceMixin(_MixinBase):
    """Flat-indexed root over several same-typed sequence datasets."""

    _is_root: bool = False
    # Single-sequence instances expose their directory (subclass contract).
    _sequence_dir: Path
    # Root instances hold one single-sequence instance of the concrete class
    # (built by the _init_root factory) per sequence directory.
    _sequences: list[Any]

    # ------------------------------------------------------------------ build

    def _init_root(
        self,
        root: str | Path,
        seq_dirs: list[Path],
        make_sequence: Callable[[Path], RootSequenceMixin],
        *,
        build_index: bool = True,
    ) -> None:
        """Populate the root from *seq_dirs*, one sub-dataset per directory.

        Args:
            root: The dataset root directory.
            seq_dirs: Sequence directories, in load order.
            make_sequence: Factory building one single-sequence instance of the
                concrete dataset class from a sequence directory.
            build_index: Build the flat index now.  Pass ``False`` for lazy
                datasets whose sequences have no keys loaded yet (the index is
                built later, when ``keys`` is set).
        """
        self._is_root = True
        self._root_dir = Path(root)
        self._sequences = [make_sequence(d) for d in seq_dirs]
        if build_index:
            self._build_flat_index()

    def _build_flat_index(self) -> None:
        lengths = [len(s) for s in self._sequences]
        self._cumulative_lengths = np.array([0, *np.cumsum(lengths)], dtype=np.intp)

    def _locate(self, idx: int) -> tuple[int, int]:
        """Map a global frame index to ``(sequence index, local row)``.

        Root datasets only. Shared by ``_load``, ``frame_info`` and
        ``derived_path`` so the flat-index arithmetic lives in one place.
        """
        if not hasattr(self, "_cumulative_lengths"):
            raise RuntimeError("No keys loaded. Set ds.keys = [...] first.")
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        seq_idx = int(np.searchsorted(self._cumulative_lengths[1:], idx, side="right"))
        return seq_idx, idx - int(self._cumulative_lengths[seq_idx])

    # ------------------------------------------------------------ subclass hooks

    def _single_available(self) -> frozenset:
        """Channels available in a single sequence -- implemented by subclasses."""
        raise NotImplementedError

    def _set_single_keys(self, keys) -> None:
        """Apply *keys* to a single sequence -- implemented by subclasses."""
        raise NotImplementedError

    # ------------------------------------------------------------- public API

    @property
    def root_dir(self) -> Path:
        return self._root_dir if self._is_root else self._sequence_dir

    @property
    def calibration(self) -> Calibration:
        """Static extrinsics from ``.apairo/calibration.yaml`` (e.g. written from
        ``/tf_static``). On a root, sequences' tables are merged -- each sequence
        carries its own, so the calibration follows the data, not the root."""
        if not self._is_root:
            return read_calibration(self._sequence_dir)
        merged = Calibration()
        for seq in self._sequences:
            cal = read_calibration(seq._sequence_dir)
            merged.update(cal)
            merged.cameras.update(cal.cameras)
        return merged

    @property
    def available(self) -> frozenset:
        """Channels available -- intersection across sequences for a root dataset."""
        if not self._is_root:
            return self._single_available()
        if not self._sequences:
            return frozenset()
        common = frozenset(self._sequences[0].available)
        for seq in self._sequences[1:]:
            common &= frozenset(seq.available)
        return common

    @property
    def sequences(self) -> list:
        """Per-sequence datasets (root datasets only)."""
        if not self._is_root:
            raise AttributeError("'sequences' is only available on root datasets.")
        return self._sequences

    @property
    def sequence_ids(self) -> list[str]:
        """Sequence directory names, in load order (root datasets only)."""
        if not self._is_root:
            raise AttributeError("'sequence_ids' is only available on root datasets.")
        return [seq._sequence_dir.name for seq in self._sequences]

    def sequence(self, seq_id: str) -> SequenceView:
        """Return a :class:`~apairo.core.sequence_view.SequenceView` for *seq_id*."""
        if not self._is_root:
            raise AttributeError("'sequence()' is only available on root datasets.")
        from apairo.core.sequence_view import SequenceView

        for seq in self._sequences:
            if seq._sequence_dir.name == seq_id:
                return SequenceView(seq, range(len(seq)), seq_id)
        raise KeyError(f"Sequence '{seq_id}' not found. Available: {self.sequence_ids}")

    def synchronize(self, reference=None, method="previous", tolerance=None):
        """Resample onto a reference clock -- see :meth:`AbstractDataset.synchronize`.

        On a root dataset each sequence is synchronized independently (clocks are
        not comparable across recordings) and the results concatenated, so an
        external clock array is only valid on a single sequence.
        """
        if not self._is_root:
            return super().synchronize(
                reference=reference, method=method, tolerance=tolerance
            )
        if reference is not None and not isinstance(reference, str):
            raise ValueError(
                "An external clock array cannot be applied to a root dataset: each "
                "sequence has its own time base. Synchronize sequences individually "
                "(ds.sequences[i].synchronize(...)) and concat the results."
            )
        from apairo.dataset.concat import ConcatDataset

        return ConcatDataset(
            [
                seq.synchronize(reference=reference, method=method, tolerance=tolerance)
                for seq in self._sequences
            ]
        )

    # ------------------------------------------------------------------ keys

    @property
    def keys(self) -> list[str]:
        if self._is_root:
            return self._sequences[0].keys if self._sequences else []
        return super().keys

    @keys.setter
    def keys(self, keys) -> None:
        if not self._is_root:
            self._set_single_keys(keys)
            return
        if keys == "all":
            keys = sorted(self.available)
        for seq in self._sequences:
            seq.keys = list(keys)
        self._build_flat_index()

    # ------------------------------------------------------------------ dunder

    def __len__(self) -> int:
        if not self._is_root:
            return super().__len__()  # type: ignore[safe-super]  # layout base implements it
        if not hasattr(self, "_cumulative_lengths"):
            raise RuntimeError("No keys loaded. Set ds.keys = [...] first.")
        return int(self._cumulative_lengths[-1])

    def _load(self, idx):
        if isinstance(idx, tuple):
            seq_id, local_idx = idx
            view = self.sequence(seq_id)
            return self._load(view._indices[local_idx])
        if not self._is_root:
            return super()._load(idx)
        seq_idx, local_idx = self._locate(idx)
        return self._sequences[seq_idx]._load(local_idx)

    # ------------------------------------------------------ frame provenance

    def frame_info(self, idx: int) -> FrameRef:
        """Channel + row each event came from, plus the sub-sequence it belongs
        to (root datasets). See :meth:`AbstractDataset.frame_info`."""
        if not self._is_root:
            return super().frame_info(idx)
        seq_idx, local_idx = self._locate(idx)
        return (
            self._sequences[seq_idx]
            .frame_info(local_idx)
            ._replace(sequence=self.sequence_ids[seq_idx])
        )

    @property
    def frame_sequence_ids(self) -> np.ndarray:
        """Sequence id per global frame index (object array). On a root, the
        sub-sequence each frame belongs to; on a single sequence, delegated."""
        if not self._is_root:
            return super().frame_sequence_ids
        result = np.empty(len(self), dtype=object)
        for seq_idx in range(len(self._sequences)):
            a = int(self._cumulative_lengths[seq_idx])
            b = int(self._cumulative_lengths[seq_idx + 1])
            result[a:b] = self.sequence_ids[seq_idx]
        return result

    @property
    def frame_stems(self) -> np.ndarray:
        """Filename stem per global frame index, concatenated over sequences."""
        if not self._is_root:
            return super().frame_stems
        if not self._sequences:
            return np.empty(0, dtype=object)
        return np.concatenate([s.frame_stems for s in self._sequences])
