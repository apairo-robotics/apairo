"""Apairo -- unified robotics dataset loader."""

import logging
from importlib.metadata import PackageNotFoundError, version as _pkg_version

from apairo.core.sample import Sample
from apairo.core.synchronous_dataset import SynchronousDataset
from apairo.core.configurable_dataset import ConfigurableDataset
from apairo.preprocess import FramePreprocessor, SequencePreprocessor

from apairo.dataset.raw import RawDataset
from apairo.dataset.tartan_kitti import TartanKittiDataset
from apairo.dataset.concat import ConcatDataset
from apairo.dataset.zip import ZipDataset
from apairo.dataset.stream import StreamDataset
from apairo.dataset import split_sequences
from apairo.core.sequence_view import SequenceView
from apairo.core.filtered_view import FilteredView
from apairo.core.channel_view import ChannelView
from apairo.core.cached_dataset import CachedDataset
from apairo.core.synchronized_view import SynchronizedView
from apairo.core.window_view import WindowView
from apairo.core.interpolator import Interpolator
from apairo.dataset.semantic_kitti import SemanticKittiDataset
from apairo.dataset.rellis import Rellis3DDataset
from apairo.dataset.goose import Goose3DDataset

from apairo.core.layout import ChannelSpec, DatasetLayout
from apairo.core.transform import Compose
from apairo.core.config import (
    register_channel,
    register_raw_channel,
    remove_channel,
    set_alias,
    verify_calibration,
    verify_config,
    verify_manifest,
)
from apairo.writer import WRITERS, ChannelWriter
from apairo.loader import DERIVED_LOADERS

logging.getLogger(__name__).addHandler(logging.NullHandler())

try:
    __version__ = _pkg_version("apairo")
except PackageNotFoundError:  # running from a source tree with no installed dist
    __version__ = "0.0.0+unknown"

__all__ = [
    "Sample",
    "SynchronousDataset",
    "ConfigurableDataset",
    "FramePreprocessor",
    "SequencePreprocessor",
    "RawDataset",
    "TartanKittiDataset",
    "ConcatDataset",
    "ZipDataset",
    "StreamDataset",
    "split_sequences",
    "SequenceView",
    "FilteredView",
    "ChannelView",
    "CachedDataset",
    "SynchronizedView",
    "WindowView",
    "Interpolator",
    "SemanticKittiDataset",
    "Rellis3DDataset",
    "Goose3DDataset",
    "ChannelSpec",
    "DatasetLayout",
    "Compose",
    "register_channel",
    "register_raw_channel",
    "remove_channel",
    "set_alias",
    "verify_config",
    "verify_manifest",
    "verify_calibration",
    "WRITERS",
    "ChannelWriter",
    "DERIVED_LOADERS",
    "__version__",
]
