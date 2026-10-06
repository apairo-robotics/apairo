# API Reference

## Synchronous datasets

### ProfiledDataset

::: apairo.core.profiled_dataset.ProfiledDataset

---

### SynchronousDataset

::: apairo.core.synchronous_dataset.SynchronousDataset

---

### SemanticKittiDataset

::: apairo.dataset.semantic_kitti.SemanticKittiDataset

---

### Goose3DDataset

::: apairo.dataset.goose.Goose3DDataset

---

### Rellis3DDataset

::: apairo.dataset.rellis.Rellis3DDataset

---

## Asynchronous datasets

### RawDataset

::: apairo.dataset.raw.RawDataset

---

### TartanKittiDataset

::: apairo.dataset.tartan_kitti.TartanKittiDataset

---

### AsyncLayoutDataset

Abstract per-channel layout base for the asynchronous family. Internal base
class — subclass it to add a fixed-channel async dataset; end users load through
`RawDataset` or `TartanKittiDataset`.

::: apairo.dataset.async_layout.AsyncLayoutDataset

---

### StreamDataset

::: apairo.dataset.stream.StreamDataset

---

### SynchronizedView

::: apairo.core.synchronized_view.SynchronizedView

---

### Interpolator

::: apairo.core.interpolator.Interpolator

---

## Dataset composition

### ConcatDataset

::: apairo.dataset.concat.ConcatDataset

---

### split_sequences

::: apairo.dataset.split_sequences

---

## Extensibility

### ConfigurableDataset

::: apairo.core.configurable_dataset.ConfigurableDataset

---

### RootSequenceMixin

Shared single-sequence vs. dataset-root handling for the asynchronous family
(flat indexing, per-sequence access, per-sequence `synchronize` + concat).
Reused by `RawDataset` and `TartanKittiDataset`.

::: apairo.core.root_sequence.RootSequenceMixin

---

### register_channel

::: apairo.core.config.register_channel

---

### Format

The format contract: one storage format of a channel. A new format is a
plugin -- see [Write a Format Plugin](../datasets/format-plugins.md).

::: apairo.core.formats.Format

---

### register_format

::: apairo.core.formats.register_format

---

### check_format

::: apairo.testing.check_format

---

### WRITERS

File writers used by dataset layouts (`DatasetLayout` export: Zarr stores, tar
archives of images). Preprocess outputs are written through the format
contract instead -- see `Format.write_frame` / `write_channel`.

```python
from apairo import WRITERS

writer = WRITERS["npy"]()
writer.write(my_array, Path("/data/output/000000.npy"))
```

::: apairo.writer.WRITERS

---

### DERIVED_LOADERS

File-level loaders for derived/preprocessed keys.
Keyed by loader name (`"npy"`, `"bin"`, `"img"`).
Each entry is a `Callable[[Path], np.ndarray]`.

```python
from apairo import DERIVED_LOADERS

tensor = DERIVED_LOADERS["npy"](Path("/data/output/000000.npy"))
```

::: apairo.loader.DERIVED_LOADERS

---

## Preprocessing

### FramePreprocessor

::: apairo.core.preprocessor.FramePreprocessor

---

### SequencePreprocessor

::: apairo.core.preprocessor.SequencePreprocessor

---

### Convert

::: apairo.preprocess.convert.Convert

---

## Data structures

### Sample

::: apairo.core.sample.Sample

---

### FrameRef

Returned by `AbstractDataset.frame_info(idx)` -- where a global frame index comes
from `(sequence, channel, row)`.

::: apairo.core.abstract_dataset.FrameRef

---

### ModalitySpec

::: apairo.core.profiled_dataset.ModalitySpec
