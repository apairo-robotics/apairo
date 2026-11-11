# Guide: HDF5 containers

Many robot recordings are stored as **containers**: one HDF5 file holding a tree
of named arrays. apairo reads them in place with the `hdf5` format
(`pip install apairo[hdf5]`). A **channel is one array**, named by its path
inside the file, and its **clock is another array** of the same file, which is
how asynchronous recordings store one clock per sensor.

## A channel is an array

```yaml
version: 1
channels:
  force:
    loader: hdf5
    directory: "."              # the file sits in the sequence directory
    array_file: "*.h5"          # a name, or a glob that matches one file
    array: /robot_state/measured_force
    key: {array: /timestamps/measured_force, units: [s]}
```

- `array` is the path of the channel's array. Frame `i` is row `i`.
- `key: {array: ...}` is the clock: an array of one timestamp per row, with
  `units` (`s`, `ms`, `us`, `ns`) or `scale` as for any key. A clock whose
  length differs from its channel's is refused, by name.
- `array_file` names the file. It may be omitted when the directory holds a
  single HDF5 file.

The format is **declared, never detected**. A container holds a whole recording
rather than one channel, and only a declaration can split it into channels, so
`apairo init` does not guess them. A directory with an `apairo.yaml` is still a
sequence of its root.

A synchronous container (one row per step for every array, no clock arrays)
borrows a clock: `timestamps_from` another channel, or a `timestamps.txt`.

## Example: REASSEMBLE

[REASSEMBLE](https://researchdata.tuwien.ac.at/records/0ewrv-8cb44) (TU Wien,
CC BY 4.0) records contact-rich assembly on a Franka arm. Each demonstration is
one HDF5 file in which every sensor has its own clock under `/timestamps`:

```
2025-01-11-14-43-37.h5
  robot_state/measured_force     (23240, 3)     ~1.7 kHz
  robot_state/joint_positions    (22549, 7)     ~1 kHz
  robot_state/pose               (13401, 7)     ~600 Hz
  timestamps/measured_force      (23240,)       epoch seconds
  timestamps/...
  hama1, hama2, hand             cameras: MP4 bytes, 30 Hz
  timestamps/hama1               (697,)         one stamp per video frame
```

A declaration ships with apairo,
[`reassemble`](https://github.com/apairo-robotics/apairo/blob/main/apairo/dataset/declarations/reassemble.yaml):
force, torque, joints, pose, gripper, and the three cameras.
`pip install apairo[reassemble]` brings `h5py` and PyAV. A sequence is a directory holding one demonstration. The
download holds every demonstration in one folder, so give each one a directory
of its own, with a link rather than a copy:

```bash
for f in data/*.h5; do d=demos/$(basename "$f" .h5); mkdir -p "$d"; ln -s "$PWD/$f" "$d/"; done
```

```console
$ apairo status demos/2025-01-11-14-43-37 --declare reassemble
channel  kind  loader  frames  rate       span         shape
force    raw   hdf5    23240   1000.0 Hz  0.00-23.24s  (3) float64
gripper  raw   hdf5    22549   970.6 Hz   0.01-23.24s  (2) float64
joints   raw   hdf5    22549   970.6 Hz   0.01-23.24s  (7) float64
pose     raw   hdf5    13401   576.6 Hz   0.00-23.24s  (7) float64
torque   raw   hdf5    23240   1000.0 Hz  0.00-23.24s  (3) float64
hama1    raw   hdf5    697     30.0 Hz    0.01-23.21s  (480, 640, 3) uint8
...
```

The rate `status` prints is frames over span. The force sensor bursts, so its
median interval is shorter than that average.

```python
import numpy as np
import apairo
from apairo_transform.interp import Se3Interp

demo = apairo.RawDataset(
    "demos/2025-01-11-14-43-37", declare="reassemble", keys=["force", "joints", "pose"]
)
t0, t1 = demo.timestamps["pose"][[0, -1]]
frames = demo.synchronize(
    reference=np.arange(t0, t1, 1 / 30),          # the cameras' 30 Hz
    method={"force": "nearest", "joints": "nearest", "pose": Se3Interp()},
    tolerance=0.01,
)
frames[100].data["force"]                         # one row, read from the file
```

Nothing is converted: rows are read from the file when a frame is accessed.

## Encoded video inside a container

The cameras are MP4 videos stored as single values of the file. `decode: video`
reads such a value as a channel of frames, decoded on access:

```yaml
  hand:
    loader: hdf5
    directory: "."
    array_file: "*.h5"
    array: /hand
    decode: video
    key: {array: /timestamps/hand, units: [s]}
```

Frame `i` is the video's `i`-th frame, `(H, W, 3)` uint8 RGB, and the clock is
REASSEMBLE's own, one stamp per frame. Reading frames in order decodes each
once; a jump decodes from the nearest keyframe before it. On the camera clock,
every frame gets the force, joints and pose of its instant:

```python
demo = apairo.RawDataset(
    "demos/2025-01-11-14-43-37",
    declare="reassemble",
    keys=["hand", "force", "joints", "pose"],  # a channel not listed below
)                                              # would be matched "previous"
frames = demo.synchronize(
    reference="hand",
    method={"force": "nearest", "joints": "nearest", "pose": Se3Interp()},
    tolerance=0.01,
)
frames[200].data["hand"].shape                    # (480, 640, 3)
```

A video stored as a file of its own (`.mp4`, `.mkv`, ...) is the `video`
format: a channel directory holding one video file, detected like any other,
with `key: {pts: true}` for the clock its presentation timestamps give.

## Notes

- `h5py` handles do not survive a `fork`. Each process opens its own, lazily,
  so a PyTorch `DataLoader` with workers is safe.
- A container is read-only. A preprocess writes its output channel in the
  sequence directory beside the file, like any derived channel.
