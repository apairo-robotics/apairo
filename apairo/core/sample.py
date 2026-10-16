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

    ``sample[key]`` is ``sample.data[key]``, and ``key in sample`` tests
    ``sample.data``. Only channels are subscriptable: channel names are free,
    so ``sample["timestamp"]`` is a channel named ``timestamp``, never the
    sample's time.
    """

    data: dict[str, Any]
    timestamp: float | None = None

    def __getitem__(self, key: str) -> Any:
        try:
            return self.data[key]
        except KeyError:
            hint = (
                "; the sample's time is `sample.timestamp`"
                if key == "timestamp"
                else ""
            )
            raise KeyError(
                f"no channel {key!r} in this sample (channels: {list(self.data)}){hint}"
            ) from None

    def __contains__(self, key: object) -> bool:
        return key in self.data

    # Not a Mapping: iterating would otherwise fall back to ``sample[0]`` and
    # fail with a KeyError, and registering as a Mapping would let
    # ``torch``'s ``default_collate`` rebuild it and drop the timestamp.
    __iter__ = None
