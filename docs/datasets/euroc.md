# Guide: EuRoC MAV

The [EuRoC MAV dataset](https://projects.asl.ethz.ch/datasets/euroc-mav) records
a micro aerial vehicle's stereo cameras and IMU, an external tracker and a fused
ground-truth state. Every sensor keeps its own nanosecond clock. apairo reads a
sequence in ASL format as downloaded, from one declaration: no subclass, no
conversion, and none of the dataset's files is modified.

## The layout

```
V1_01_easy/mav0/
  cam0/data/   1403715273262142976.png  ...    # 20 Hz, named by their stamp (ns)
  cam1/data/   1403715273262142976.png  ...    # 20 Hz
  imu0/data.csv                                # 200 Hz
  vicon0/data.csv                              # 100 Hz  (Vicon Room: V1_*, V2_*)
  leica0/data.csv                              # ~15 Hz  (Machine Hall: MH_*)
  state_groundtruth_estimate0/data.csv         # 200 Hz
```

Each `data.csv` is comma-separated, with the timestamp (nanoseconds) in its first
column and one header line that carries units:

```
#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],...
1403715273262142976,-0.0020943951023931952,0.017453292519943295,0.07749261878854824,9.0874956666666655,...
```

## The declaration

```yaml
version: 1
channels:
  cam0: {loader: img, directory: cam0/data, key: {name: '(\d+)', units: [ns]}}
  cam1: {loader: img, directory: cam1/data, key: {name: '(\d+)', units: [ns]}}
  imu0: {loader: csv, key: {column: timestamp, units: [ns]}}
  vicon0: {loader: csv, key: {column: timestamp, units: [ns]}}
  state_groundtruth_estimate0:
    loader: csv
    alias: groundtruth
    key: {column: timestamp, units: [ns]}
```

- The images live one level down, so `directory: cam0/data` points at them; their
  clock is the filename, a nanosecond epoch folded to seconds by `units: [ns]`.
- `loader: csv` reads each table as one frame per row.
  `key: {column: timestamp, units: [ns]}` takes the clock from the first column.
  The loader drops the leading `#` and the trailing `[unit]` from the header, so
  the column is `timestamp` and the IMU's fields are `w_RS_S_x`, `a_RS_S_x`, …
  An integer nanosecond stamp is converted exactly before it is scaled.
- `alias: groundtruth` gives the long directory name a short public one.

The external tracker depends on the hall, so two declarations ship with apairo:
[`euroc_vicon_room`](https://github.com/apairo-robotics/apairo/blob/main/apairo/dataset/declarations/euroc_vicon_room.yaml) (`vicon0`, shown above) and
[`euroc_machine_hall`](https://github.com/apairo-robotics/apairo/blob/main/apairo/dataset/declarations/euroc_machine_hall.yaml) (`leica0`).
`pip install apairo[euroc]` brings what reading the images needs. Pass the
right one by name with `declare=` or `--declare`, or copy it to
`<sequence>/mav0/apairo.yaml` to edit it:

```python
from apairo import RawDataset

ds = RawDataset("V1_01_easy/mav0", declare="euroc_vicon_room")
ds.shape
# {'cam0': (480, 752), 'cam1': (480, 752), 'groundtruth': (16,), 'imu0': (6,), 'vicon0': (7,)}
```

On first open apairo writes its registry, a `.apairo/` directory, next to the
data; on a read-only mount it keeps it in memory instead.

```console
$ apairo status V1_01_easy/mav0 --declare euroc_vicon_room
RawDataset - mav0   (sequence)
----------------------------------------------------
start       1403715271.71s   (span shown relative to this)
channel                                    kind  loader  frames  rate      span          shape
cam0                                       raw   img     2912    20.0 Hz   1.56-147.11s  (480, 752) uint8
cam1                                       raw   img     2912    20.0 Hz   1.56-147.11s  (480, 752) uint8
imu0                                       raw   csv     29120   200.0 Hz  1.56-147.15s  (6) float64
groundtruth (state_groundtruth_estimate0)  raw   csv     28712   200.0 Hz  2.60-146.15s  (16) float64
vicon0                                     raw   csv     14629   100.0 Hz  0.00-146.28s  (7) float64
events      78285
issues      none
```

## Putting the IMU and the state on the camera clock

```python
vio = RawDataset("V1_01_easy/mav0", keys=["cam0", "cam1", "imu0", "groundtruth"],
                 declare="euroc_vicon_room")
frames = vio.synchronize(reference="cam0", method="nearest", tolerance=0.005)

sample = frames[0]
sample.data["cam0"], sample.data["cam1"]   # the stereo pair, (480, 752) each
sample.data["imu0"]                        # w_RS_S_xyz, a_RS_S_xyz
sample.data["groundtruth"][:3]             # position
```

The status above shows that the ground truth starts a second after the cameras
and stops a second before them. With a tolerance, the camera frames outside it
are dropped instead of being paired with a state from another moment: on
`V1_01_easy`, 2872 of the 2912 frames remain. Without a tolerance those 40 frames
would silently carry the first or the last state, up to 1.04 s away.

`time_offsets()` gives the distance of every match:

```python
abs(frames.time_offsets("imu0")).max()         # 0.0: IMU and camera stamps coincide
abs(frames.time_offsets("groundtruth")).max()  # 0.005: never beyond the tolerance
```

!!! note "A tracker with gaps"
    Add `vicon0` to the keys and 44 more frames drop, all *inside* the recording:
    the motion capture runs at 100 Hz but has gaps of up to 18 ms, so some camera
    frames have no pose within 5 ms. The tolerance is what makes that visible.

## Run it

[`examples/euroc_synchronize.py`](https://github.com/apairo-robotics/apairo/blob/main/examples/euroc_synchronize.py)
does all of the above, and picks the declaration from the tracker directory it
finds:

```bash
# https://doi.org/10.3929/ethz-b-000690084 -> vicon_room1.zip -> V1_01_easy (ASL format)
APAIRO_EUROC_SEQ=/data/euroc/V1_01_easy/mav0 python examples/euroc_synchronize.py
```

!!! note "What was checked"
    Both declarations and the figures on this page were checked against real
    sequences: `V1_01_easy` (Vicon Room, 2912 camera frames) and
    `MH_05_difficult` (Machine Hall, 2273 camera frames, of which 2221 lie inside
    the ground truth). CI runs the same example on synthetic miniatures of both
    layouts (`test/assets/mini_euroc`), which hold no EuRoC data.
