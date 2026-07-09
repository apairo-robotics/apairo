"""ChannelWriter -- write a per-frame channel in the on-disk format apairo reads.

External producers (annotation tools, or preprocessors that already *hold* the
data rather than expose a deterministic ``process(sample)`` function) use this
instead of hand-writing the layout.  It owns the three things that make a channel
loadable, so they stay in apairo rather than drifting in every tool:

1. the **frame-naming policy** the loader reads back (no ``_`` in a stem),
2. a ``timestamps.txt`` kept in the frame order the loader sorts to,
3. **registration** in ``.apairo/channels.yaml`` on :meth:`close`.

Deterministic derived channels still belong in
:meth:`~apairo.core.configurable_dataset.ConfigurableDataset.run_preprocess`;
this is the path for data produced *outside* apairo (e.g. a labeling tool).

Usage::

    with apairo.ChannelWriter(seq_dir, "ground_truth", loader="npys",
                              timestamps_from="ouster_points",
                              sources=["ouster_points"]) as w:
        w.add(labels, stem="001813", timestamp=t)   # -> seq/ground_truth/001813.npy
    # channels.yaml now declares ground_truth (kind: preprocess)
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from apairo.core.config import register_channel
from apairo.core.naming import frame_stem_is_valid, require_frame_stem
from apairo.writer.bin_writer import BINWriter
from apairo.writer.npy_writer import NPYWriter

# Per-frame array loaders this writer can emit (one data file per frame). The
# stacked ``npy`` (one file, many rows) and ``img``/``zarr`` use different on-disk
# models and are out of scope here.
_PER_FRAME: dict[str, tuple[Callable[[], Any], str]] = {
    "npys": (NPYWriter, ".npy"),
    "bin": (BINWriter, ".bin"),
}


class ChannelWriter:
    """Write per-frame frames into ``<seq_dir>/<channel>/`` and register them.

    Args:
        seq_dir: The sequence directory the channel lives in.
        channel: Channel name (its on-disk sub-directory and registry key).
        loader: Per-frame loader to write for -- ``"npys"`` (``.npy``, any dtype)
            or ``"bin"`` (raw ``float32``).
        timestamps_from: Channel whose timestamps this one shares (provenance).
        sources: Channels this one is derived from (provenance).
        frame: Coordinate frame of the data (descriptive metadata).
    """

    def __init__(
        self,
        seq_dir,
        channel: str,
        *,
        loader: str = "npys",
        timestamps_from: str | None = None,
        sources: list[str] | None = None,
        frame: str | None = None,
    ) -> None:
        if loader not in _PER_FRAME:
            raise ValueError(
                f"ChannelWriter handles per-frame loaders {sorted(_PER_FRAME)}, "
                f"not {loader!r} (stacked 'npy'/'img'/'zarr' are out of scope)."
            )
        writer_cls, self._ext = _PER_FRAME[loader]
        self._seq_dir = Path(seq_dir)
        self._channel = channel
        self._loader = loader
        self._timestamps_from = timestamps_from
        self._sources = list(sources) if sources else None
        self._frame = frame
        self._cdir = self._seq_dir / channel
        self._writer = writer_cls()
        self._closed = False
        # Resume an existing channel so incremental annotation across runs stays
        # consistent: pick up the frames already on disk and their timestamps.
        self._ts: dict[str, float | None] = self._read_existing()

    def _read_existing(self) -> dict[str, float | None]:
        if not self._cdir.is_dir():
            return {}
        stems = sorted(
            p.stem
            for p in self._cdir.glob(f"*{self._ext}")
            if frame_stem_is_valid(p.stem)
        )
        if not stems:
            return {}
        ts_path = self._cdir / "timestamps.txt"
        if ts_path.exists():
            rows = np.atleast_1d(np.loadtxt(ts_path)).tolist()
            if len(rows) == len(stems):
                return dict(zip(stems, rows, strict=True))
        return {s: None for s in stems}

    def add(self, data, stem, timestamp: float | None = None) -> ChannelWriter:
        """Write one frame as ``<seq>/<channel>/<stem><ext>`` and record its
        timestamp.  *stem* is the frame id (no extension) and must not contain
        ``_``.  Returns ``self`` for chaining."""
        if self._closed:
            raise RuntimeError("ChannelWriter is closed.")
        stem = require_frame_stem(str(stem))
        self._cdir.mkdir(parents=True, exist_ok=True)
        self._writer.write(np.asarray(data), self._cdir / f"{stem}{self._ext}")
        self._ts[stem] = None if timestamp is None else float(timestamp)
        return self

    def close(self) -> None:
        """Commit: write ``timestamps.txt`` in the loader's frame order and
        register the channel in ``.apairo/channels.yaml``.  Idempotent."""
        if self._closed:
            return
        if not self._ts:
            raise RuntimeError("ChannelWriter.close() with no frames written.")
        # Frames load in sorted-stem order; timestamps.txt must follow it row-for-row.
        stems = sorted(self._ts)
        have = [self._ts[s] is not None for s in stems]
        if any(have) and not all(have):
            missing = [s for s in stems if self._ts[s] is None]
            raise ValueError(
                "timestamps must be given for all frames or none; "
                f"missing for {missing}."
            )
        if all(have):
            np.savetxt(
                self._cdir / "timestamps.txt", np.array([self._ts[s] for s in stems])
            )
        register_channel(
            self._seq_dir,
            self._channel,
            self._loader,
            timestamps_from=self._timestamps_from,
            sources=self._sources,
            frame=self._frame,
        )
        self._closed = True

    def __enter__(self) -> ChannelWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Commit only on a clean exit; a failed run leaves the written frames on
        # disk but unregistered (a later run resumes them, or `apairo init` finds
        # them), rather than registering a half-written channel.
        if exc_type is None:
            self.close()
