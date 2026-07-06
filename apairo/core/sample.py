from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Sample:
    """A single dataset sample -- one timeline event or one complete frame.

    ``timestamp`` follows the frame's *clock*, not merely whether the data is
    synchronous:

    - Asynchronous event -- ``data`` has one key; ``timestamp`` is that event's.
    - Synchronous *clocked* frame (a ``synchronize()`` result) -- ``data`` has
      all requested keys; ``timestamp`` is the reference-clock tick the frame
      was resampled onto.
    - Synchronous *clockless* frame (a profiled dataset) -- ``data`` has all
      keys; ``timestamp`` is ``None``.
    """

    data: dict[str, Any]
    timestamp: float | None = None
