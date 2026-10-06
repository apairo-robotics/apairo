"""A format conversion is a preprocess.

Copying a channel into another format is a derived channel like any other: it
is written by the runner, registered with its provenance (``sources``, the
recipe), stamped with the source's clock, and protected against an accidental
overwrite. Nothing about it is special-cased::

    RawDataset.run_preprocess(Convert("velodyne_0", to="zarr"), seq)
    # -> seq/velodyne_0_zarr/, registered as a zarr channel
"""

from __future__ import annotations

from typing import Any

from apairo.core.preprocessor import FramePreprocessor
from apairo.core.sample import Sample


class Convert(FramePreprocessor):
    """Copy channel *channel* into the format *to*, as channel *output_key*
    (``<channel>_<to>`` by default). Any registered format that writes is a
    target, a plugin's included; the runner refuses one that cannot hold the
    frames, by name."""

    def __init__(self, channel: str, to: str, *, output_key: str | None = None) -> None:
        # Instance attributes, read by the runner like a subclass's declarations.
        self.input_keys = [channel]  # type: ignore[misc]
        self.output_key = output_key or f"{channel}_{to}"  # type: ignore[misc]
        self.output_loader = to  # type: ignore[misc]
        self.sources = [channel]  # type: ignore[misc]
        self.channel = channel

    def __call__(self, sample: Sample) -> Any:
        return sample.data[self.channel]
