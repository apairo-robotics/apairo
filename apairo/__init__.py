"""Apairo — unified robotics dataset loader."""

from apairo.core.sample import Sample
from apairo.core.synchronous_dataset import SynchronousDataset
from apairo.core.configurable_dataset import ConfigurableDataset
from apairo.preprocess import FramePreprocessor, SequencePreprocessor

from apairo.dataset.kitti import KittiDataset
from apairo.dataset.tartan_kitti import TartanKittiDataset
from apairo.dataset.concat import ConcatDataset, TorchConcatDataset
from apairo.dataset.torch_wrappers import TorchKittiDataset, TorchKittiIterDataset
from apairo.dataset import split_sequences
from apairo.dataset.semantic_kitti import SemanticKittiDataset
from apairo.dataset.rellis import Rellis3DDataset
from apairo.dataset.goose import Goose3DDataset

from apairo.core.config import register_channel
from apairo.writer import WRITERS
from apairo.loader import DERIVED_LOADERS

from apairo.sampler.low_freq_uniform_sampler import LowFreqUniformSampler
from apairo.sampler.latest_sync_sampler import LatestSyncSampler

__version__ = "0.1.0"

__all__ = [
    "Sample",
    "SynchronousDataset",
    "ConfigurableDataset",
    "FramePreprocessor",
    "SequencePreprocessor",
    "KittiDataset",
    "TartanKittiDataset",
    "ConcatDataset",
    "TorchConcatDataset",
    "TorchKittiDataset",
    "TorchKittiIterDataset",
    "split_sequences",
    "SemanticKittiDataset",
    "Rellis3DDataset",
    "Goose3DDataset",
    "register_channel",
    "WRITERS",
    "DERIVED_LOADERS",
    "LowFreqUniformSampler",
    "LatestSyncSampler",
    "__version__",
]
