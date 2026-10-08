# Guide: KUKA F/T and IMU

This dataset logs three sensors on a KUKA LBR Med 14 arm (A. Skrede, 2024,
[Zenodo 10.5281/zenodo.11096791](https://doi.org/10.5281/zenodo.11096791),
CC BY 4.0):

- the orientation of the force/torque sensor, from the robot controller;
- the wrench measured by the force/torque sensor (ATI Gamma);
- the acceleration of an IMU (MPU6886) on the end effector.

Each sensor writes its own CSV file on its own clock, at its own rate. That is
the shape of most robot-arm data, and where alignment matters most. One of the
sensors also stamps its readings late. This guide reads the logs in place,
measures that delay in the data, corrects it with a declaration, and puts every
reading on the robot's clock.

## The layout

The record is one folder of CSV files. Three runs repeat the same motion, a
rotation about the sensor's y axis: `1-baseline`, `2-vibrations` (gentle taps
with a rubber hammer) and `3-vibrations-contact` (taps, and a hand on the end
effector).

```
<run>_orientations.csv   t, r11..r33   the sensor's orientation from the robot   ~100 Hz
<run>_wrench.csv         t, fx..tz     the force/torque sensor                   ~700 Hz
<run>_accel.csv          t, ax, ay, az the IMU on the end effector               ~254 Hz
0-calibration_fts-accel.csv, 0-steady-state_accel.csv, 0-steady-state_wrench.csv
```

`t` is a microsecond epoch. The three `0-*` tables have no `t` column: they
hold calibration poses and the sensors at rest, not time series.

## The declaration

```yaml
--8<-- "apairo/dataset/declarations/kuka_ft_imu.yaml"
```

- `directory: "."` and `array_file` reach each table in the folder itself.
- `key: {column: t, units: [us]}` takes each table's clock from its `t`
  column and keeps that column out of the frame data.
- `latency: 0.008416` is the IMU's delay, from the dataset's description; see
  below.

The declaration ships with apairo as `kuka_ft_imu`. Pass it by name:

```python
from apairo import RawDataset

ds = RawDataset("kuka_logs", declare="kuka_ft_imu",
                keys=["1-baseline_orientations", "1-baseline_wrench", "1-baseline_accel"])
```

A dataset is opened on the channels of one run. The `0-*` tables have no clock,
so `apairo` reports them and does not load them:

```console
$ apairo status kuka_logs --declare kuka_ft_imu
RawDataset - kuka_logs   (sequence)
----------------------------------------------------
start       1708857503.92s   (span shown relative to this)
channel                            kind  loader  frames  rate      span                  shape
0-calibration_fts-accel            raw   csv     24      -         -                     (21) float64
0-steady-state_accel               raw   csv     256     -         -                     (3) float64
0-steady-state_wrench              raw   csv     698     -         -                     (6) float64
1-baseline_accel                   raw   csv     1593    254.1 Hz  0.00-6.26s            (3) float64
1-baseline_orientations            raw   csv     627     100.0 Hz  0.01-6.27s            (9) float64
1-baseline_wrench                  raw   csv     4376    698.1 Hz  0.01-6.27s            (6) float64
2-vibrations_accel                 raw   csv     1593    254.1 Hz  344.50-350.76s        (3) float64
...
3-vibrations-contact_wrench        raw   csv     4373    698.1 Hz  286766.24-286772.51s  (6) float64
events      20762
issues
            - channel '0-calibration_fts-accel': no clock -- ...
            - channel '0-steady-state_accel': no clock -- ...
            - channel '0-steady-state_wrench': no clock -- ...

declaration (kuka_ft_imu.yaml): ok
```

## A delay the timestamps cannot show

The authors read the IMU through an Arduino over I2C and forwarded it over
Ethernet. Their description says that this "resulted in a phase of the IMU
signal by 8416 μs": each reading reached the computer, and was stamped, about
8.4 ms after it was taken.

Comparing timestamps cannot reveal this, because they are all late together.
The data can. While the arm rotates, the IMU's acceleration and the F/T
sensor's force follow the same gravity. Shift one signal against the other
until a linear fit of one on the other is best: that shift is the delay
between the two clocks. The example does this between the IMU's `ax` and the
sensor's `fz`:

| Run | `ax`–`fz` correlation | IMU behind the F/T sensor, logged clocks | after `latency: 0.008416` |
|---|---|---|---|
| `1-baseline` | 0.995 | +7.6 ms | −0.8 ms |
| `2-vibrations` | 0.991 | +7.2 ms | −1.2 ms |
| `3-vibrations-contact` | 0.866 | not measurable | |

On the first two runs, the data shows the documented delay in sign and in
size, to within about a millisecond. That is the precision of this
measurement, on signals sampled every 1.4 to 4 ms. The F/T sensor has its own
transport delay, which is not documented, so this measures the difference
between the two delays. On the third run, the hand on the end effector pushes
on the force sensor and not on the IMU, so `fz` no longer follows gravity
alone, and the fit finds no meaningful shift.

Declaring `latency` moves the IMU's clock back by 8416 μs before anything is
aligned to it. Without it, nearest-neighbour matching pairs each robot sample
with an IMU reading taken about 8 ms earlier than its stamp says.

## Every reading on the robot's clock

```python
view = ds.synchronize(reference="1-baseline_orientations", method="nearest",
                      tolerance=0.002)

sample = view[300]
sample.data["1-baseline_orientations"]   # r11..r33, the sensor's orientation
sample.data["1-baseline_wrench"]         # fx, fy, fz, tx, ty, tz
sample.data["1-baseline_accel"]          # ax, ay, az, on the corrected clock
view.time_offsets("1-baseline_accel")    # how far each match is from the robot sample
```

All 627 orientation samples of `1-baseline` find an F/T reading (median
offset 0.36 ms) and an IMU reading (median 1.00 ms) within 2 ms. The whole
example is
[`examples/kuka_ft_imu.py`](https://github.com/apairo-robotics/apairo/blob/main/examples/kuka_ft_imu.py):

```console
$ APAIRO_KUKA_DIR=kuka_logs python examples/kuka_ft_imu.py
1-baseline_orientations      627 rows at  100.0 Hz
1-baseline_wrench           4376 rows at  698.1 Hz
1-baseline_accel            1593 rows at  254.1 Hz

IMU behind the F/T sensor, measured from the data:
  on the logged clocks       :  +7.6 ms   (documented: +8.4 ms)
  with the declared latency  :  -0.8 ms

627 of 627 orientation samples have both readings within 2 ms
  1-baseline_wrench        median offset 0.36 ms, max 0.73 ms
  1-baseline_accel         median offset 1.00 ms, max 1.99 ms
```

!!! note "What was checked"
    The declaration, the figures and the example output on this page come from
    the full record as downloaded from Zenodo (12 files, published
    2024-05-01). CI runs the example on a real excerpt
    (`test/assets/mini_kuka_ft_imu`): the `1-baseline` tables whole, and the
    first second of the two other runs.
