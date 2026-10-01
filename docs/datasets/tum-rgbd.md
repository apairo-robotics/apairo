# Guide: TUM RGB-D

The [TUM RGB-D benchmark](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)
records a Kinect's colour and depth streams, a motion-capture trajectory and the
Kinect's accelerometer. Each runs on its own clock. apairo reads a sequence as
downloaded, from one declaration: no subclass, no conversion, and none of the
dataset's files is modified.

## The layout

```
rgbd_dataset_freiburg1_xyz/
  rgb/     1305031102.175304.png  1305031102.211214.png  ...   # 30 Hz
  depth/   1305031102.160407.png  1305031102.194330.png  ...   # 30 Hz, its own stamps
  groundtruth.txt      # timestamp tx ty tz qx qy qz qw          100 Hz
  accelerometer.txt    # timestamp ax ay az                      500 Hz
  rgb.txt  depth.txt   # index files (timestamp -> filename), not needed
```

Two things carry the clocks:

- each image is named after its timestamp, `<sec>.<usec>.png`;
- each table is whitespace-separated, with `#` comment lines and the timestamp
  (seconds) in its first column. The last comment names the columns.

## The declaration

```yaml
version: 1
channels:
  rgb:   {loader: img, key: {name: '(\d+)\.(\d+)'}}
  depth: {loader: img, key: {name: '(\d+)\.(\d+)'}}
  groundtruth:
    loader: csv
    directory: "."
    array_file: groundtruth.txt
    key: {column: timestamp}
  accelerometer:
    loader: csv
    directory: "."
    array_file: accelerometer.txt
    key: {column: timestamp}
```

- `key: {name: ...}` parses each image's clock from its filename: the two capture
  groups become `<sec>.<usec>`. See [the `key` field](bring-your-own-dataset.md#the-key-field).
- `loader: csv` reads a text table as one frame per row. `directory: "."` reaches a
  table that sits at the sequence root, `array_file` names it, and
  `key: {column: timestamp}` takes the clock from the column that the header
  comment calls `timestamp`. That column is kept out of the frame data.

The file ships as
[`examples/declarations/tum_rgbd.yaml`](https://github.com/apairo-robotics/apairo/blob/main/examples/declarations/tum_rgbd.yaml).
Save it as `<sequence>/apairo.yaml`, or keep it outside the dataset and pass it
with `declare=`:

```python
from apairo import RawDataset

ds = RawDataset("rgbd_dataset_freiburg1_xyz", declare="tum_rgbd.yaml")
ds.shape
# {'accelerometer': (3,), 'depth': (480, 640), 'groundtruth': (7,), 'rgb': (480, 640, 3)}
```

On first open apairo writes its registry, a `.apairo/` directory, next to the
data; on a read-only mount it keeps it in memory instead.

```console
$ apairo status rgbd_dataset_freiburg1_xyz --declare tum_rgbd.yaml
RawDataset - rgbd_dataset_freiburg1_xyz   (sequence)
----------------------------------------------------
start       1305031098.38s   (span shown relative to this)
channel        kind  loader  frames  rate      span         shape
accelerometer  raw   csv     15158   498.8 Hz  0.00-30.39s  (3) float64
depth          raw   img     798     30.0 Hz   3.78-30.37s  (480, 640) uint16
groundtruth    raw   csv     3000    99.7 Hz   0.28-30.37s  (7) float64
rgb            raw   img     798     30.0 Hz   3.79-30.37s  (480, 640, 3) uint8
events      19754
issues      none
```

## Associating depth and pose to each colour frame

TUM ships `associate.py` to pair every colour image with the depth image closest
in time, within 20 ms. `synchronize()` is that association for every channel at
once:

```python
frames = ds.synchronize(reference="rgb", method="nearest", tolerance=0.02)

sample = frames[0]
sample.data["rgb"]           # (480, 640, 3) uint8
sample.data["depth"] / 5000  # metres: TUM depth PNGs hold 5000 units per metre
sample.data["groundtruth"]   # tx ty tz qx qy qz qw
```

A colour frame with no match within the tolerance in some channel is dropped,
not served with stale data. What each match costs is reported, per frame:

```python
abs(frames.time_offsets("depth")).max()        # 0.0172 s on freiburg1_xyz
abs(frames.time_offsets("groundtruth")).mean() # 0.0030 s
```

### How it differs from `associate.py`

`nearest` matches each colour frame on its own, so one depth image can serve two
colour frames. `associate.py` pairs one-to-one and discards the extra colour
frame. On `freiburg1_xyz`, colour against depth only:

| | Frames kept | |
|---|---|---|
| `associate.py` (20 ms) | 792 | one-to-one |
| `synchronize(nearest, 0.02)` | 798 | 789 of the 792 pairs are identical; 9 depth images serve two colour frames |

The reuse is visible, not silent: `frames.frame_indices["depth"]` gives the depth
row behind every frame, so you can see which frames share a depth image and drop
them if you need a strict one-to-one association.

## Run it

[`examples/tum_rgbd_associate.py`](https://github.com/apairo-robotics/apairo/blob/main/examples/tum_rgbd_associate.py)
does all of the above:

```bash
# https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download -> freiburg1_xyz (tgz, 0.5 GB)
APAIRO_TUM_SEQ=/data/tum/rgbd_dataset_freiburg1_xyz python examples/tum_rgbd_associate.py
```

!!! note "What was checked"
    The declaration and the figures on this page were checked against the real
    `rgbd_dataset_freiburg1_xyz` sequence (798 colour frames, 796 of them with a
    depth image, a pose and an accelerometer reading within 20 ms). CI runs the
    same example on a synthetic miniature of the layout
    (`test/assets/mini_tum`), which holds no TUM data.
