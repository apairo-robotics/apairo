# Quickstart

## Synchronous datasets

Synchronous datasets (SemanticKITTI, GOOSE, Rellis-3D) index directly by frame number. Every `ds[i]` returns a complete set of co-captured modalities.

```python
import apairo

ds = apairo.SemanticKittiDataset(
    "/data/semantic_kitti",
    keys=["lidar", "labels"],
    split="train",          # optional: filter by split directory
)

print(len(ds))              # total number of frames across all sequences
sample = ds[0]
print(sample.data["lidar"].shape)   # torch.Size([N, 4])  — x, y, z, intensity
print(sample.data["labels"].shape)  # torch.Size([N])     — semantic class IDs
print(sample.timestamp)             # None  (synchronous, no timestamps)
```

The same API works for all synchronous datasets:

```python
ds = apairo.Goose3DDataset("/data/goose", keys=["lidar", "labels"], split="train")
ds = apairo.Rellis3DDataset("/data/rellis", keys=["lidar"])
```

## Asynchronous datasets

Asynchronous datasets (TartanDrive, KITTI) interleave multiple sensor streams into a single timestamp-ordered timeline. Every `ds[i]` returns one event from one sensor.

```python
ds = apairo.TartanKittiDataset("/data/tartan/2024-01-01_forest")
print(ds.keys)              # discovered channels: ["velodyne_0", "image_left", ...]
print(len(ds))              # total events across all channels

sample = ds[42]
print(sample.timestamp)             # float  — seconds
print(list(sample.data.keys()))     # ["velodyne_0"]  — exactly one channel per event
print(sample.data["velodyne_0"].shape)
```

## The Sample object

Every `__getitem__` call returns a [`Sample`][apairo.core.sample.Sample]:

```python
@dataclass
class Sample:
    data: dict[str, torch.Tensor]   # key → tensor
    timestamp: float | None         # None for synchronous datasets
```

## Iterating

All datasets are iterable:

```python
for sample in ds:
    process(sample.data["lidar"])
```

## Selecting keys

Pass `keys=` to load only the modalities you need — useful when some channels are large or slow to load:

```python
ds = apairo.SemanticKittiDataset("/data/semantic_kitti", keys=["labels"])
```

Available keys for each dataset class are exposed as a class attribute:

```python
print(apairo.Goose3DDataset.available_keys)   # frozenset({'lidar', 'labels'})
```
