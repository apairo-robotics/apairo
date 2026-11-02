"""Frames of an encoded video, with random access: frame ``i`` is the ``i``-th
frame in presentation order, an ``(H, W, 3)`` uint8 RGB array.

A video is decoded with PyAV (``pip install apairo[video]``), imported when a
frame is first read. Its packets are indexed once, without decoding: their
presentation timestamps give the frame order and the clock, and the keyframes
are where a seek can land. Reading frame ``i`` then seeks to the last keyframe
at or before it and decodes forward; reading frame ``i + 1`` next continues the
same decoder, so a sequential pass decodes each frame once.

The source is a file or the encoded bytes themselves (a video stored inside a
container, as REASSEMBLE's cameras are in its HDF5 files). A decoder is opened
lazily, once per process: it does not survive a ``fork``.
"""

from __future__ import annotations

import io
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from apairo.core.abstract_loader import AbstractLoader

EXTENSIONS = frozenset({".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"})


def _av():
    try:
        import av
    except ImportError as exc:
        raise ImportError(
            "a video channel needs PyAV: pip install 'apairo[video]'"
        ) from exc
    return av


class VideoReader:
    """Random access to the frames of one video stream.

    Args:
        source: A path, or a callable returning the encoded bytes (called once
            per process, so bytes inside a container are read lazily).
        stream: Index of the video stream in the file.
    """

    def __init__(
        self, source: str | Path | Callable[[], bytes], stream: int = 0
    ) -> None:
        self._source = source
        self._stream_index = stream
        self._pid: int | None = None
        self._container: Any = None
        self._stream: Any = None
        self._decoder: Any = None
        self._next = -1  # index of the frame the open decoder yields next
        self._index()

    # -------------------------------------------------------------- opening

    def _open(self):
        if self._container is None or self._pid != os.getpid():
            av = _av()
            src = self._source
            target = io.BytesIO(src()) if callable(src) else str(src)
            self._container = av.open(target)
            self._stream = self._container.streams.video[self._stream_index]
            self._stream.thread_type = "AUTO"
            self._decoder = None
            self._next = -1
            self._pid = os.getpid()
        return self._container, self._stream

    def _index(self) -> None:
        container, stream = self._open()
        pts, keys = [], []
        for packet in container.demux(stream):
            if packet.pts is None:  # the flush packet
                continue
            pts.append(packet.pts)
            if packet.is_keyframe:
                keys.append(packet.pts)
        if not pts:
            raise ValueError("the video holds no frame")
        self.pts = np.sort(np.asarray(pts, dtype=np.int64))
        self._keyframes = np.sort(np.asarray(keys or [self.pts[0]], dtype=np.int64))
        self.time_base = float(stream.time_base) if stream.time_base else 1.0
        self.width, self.height = (
            stream.codec_context.width,
            stream.codec_context.height,
        )
        self._container.seek(0, stream=stream)  # demuxing consumed the file
        self._decoder = None
        self._next = -1

    # -------------------------------------------------------------- reading

    def __len__(self) -> int:
        return len(self.pts)

    def seconds(self) -> np.ndarray:
        """Presentation time of every frame, in seconds from the stream's origin."""
        return self.pts.astype(np.float64) * self.time_base

    def __getitem__(self, idx: int) -> np.ndarray:
        n = len(self.pts)
        if idx < 0:
            idx += n
        if not 0 <= idx < n:
            raise IndexError(f"frame {idx} out of range for {n} frames")
        container, stream = self._open()
        target = int(self.pts[idx])
        if self._decoder is None or idx != self._next:
            key = int(
                self._keyframes[
                    np.searchsorted(self._keyframes, target, side="right") - 1
                ]
            )
            container.seek(key, stream=stream, backward=True, any_frame=False)
            self._decoder = container.decode(stream)
        for frame in self._decoder:
            if frame.pts is None or frame.pts < target:
                continue
            if frame.pts > target:  # the requested frame was not decodable
                break
            self._next = idx + 1
            return frame.to_ndarray(format="rgb24")
        self._decoder = None
        raise ValueError(f"frame {idx} (pts {target}) could not be decoded")

    def __getstate__(self):  # pickled into DataLoader workers without a decoder
        state = self.__dict__.copy()
        state.update(_container=None, _stream=None, _decoder=None, _pid=None, _next=-1)
        return state


def locate_video(directory: Path, name: str | None) -> Path:
    """The video file of a channel: ``name`` (a name or a glob matching one
    file), else the only video file in ``directory``."""
    directory = Path(directory)
    if name:
        matches = sorted(directory.glob(name))
    else:
        matches = sorted(
            p for p in directory.iterdir() if p.suffix.lower() in EXTENSIONS
        )
    if len(matches) != 1:
        found = ", ".join(p.name for p in matches) or "none"
        what = f"'{name}' must match one file" if name else "must hold one video file"
        raise FileNotFoundError(f"{directory}: {what}; found: {found}")
    return matches[0]


class VideoLoader(AbstractLoader):
    """A channel stored as one video file: frame ``i`` is its ``i``-th frame."""

    def __init__(self, directory: str, array_file: str | None = None) -> None:
        self.directory = Path(directory)
        self.path = locate_video(self.directory, array_file)
        self.reader = VideoReader(self.path)

    def __len__(self) -> int:
        return len(self.reader)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.reader[idx]

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.reader.height, self.reader.width, 3)
