"""Read the KUKA F/T and IMU logs in place, and find the IMU's latency in the data.

Three sensors on one robot arm, each logged to its own CSV file on its own
clock: the sensor's orientation from the robot at 100 Hz, the force/torque
sensor at 700 Hz, the IMU at 254 Hz. The ``kuka_ft_imu`` declaration, which
ships with apairo, gives each table its clock (column ``t``, microseconds) and
the IMU's latency: it was read by an Arduino and forwarded over Ethernet, a
delay the dataset's authors give as 8416 us.

Timestamps alone cannot show that delay; the data can. While the arm rotates,
the IMU's acceleration and the F/T sensor's force follow the same gravity, so
the shift that best lines one up with the other is the delay between their
clocks -- measured here on the logged clocks, then on the corrected ones.
"""

import os
from pathlib import Path

import numpy as np
import yaml

from apairo import RawDataset
from apairo.dataset.registry import declaration

LOGS = Path(os.environ.get("APAIRO_KUKA_DIR", "/data/kuka_ft_imu"))
RUN = "1-baseline"
KEYS = [f"{RUN}_orientations", f"{RUN}_wrench", f"{RUN}_accel"]

ds = RawDataset(LOGS, keys=KEYS, declare="kuka_ft_imu")
for key in KEYS:
    ts = ds.timestamps[key]
    rate = (len(ts) - 1) / (ts[-1] - ts[0])
    print(f"{key:<26} {len(ts):>5} rows at {rate:6.1f} Hz")


def rows(key: str) -> np.ndarray:
    loader = ds.loaders[key]
    return np.stack([loader[i] for i in range(len(loader))])


def delay(sig_t, sig, ref_t, ref) -> float:
    """The shift d that best fits ref(t) with a * sig(t + d) + b: how much
    later the signal's events show on its clock than on the reference's."""
    grid = np.arange(
        max(sig_t[0], ref_t[0]) + 0.05, min(sig_t[-1], ref_t[-1]) - 0.05, 5e-4
    )
    target = np.interp(grid, ref_t, ref)
    shifts = np.arange(-0.03, 0.03, 2e-4)
    residuals = []
    for d in shifts:
        fit = np.column_stack([np.interp(grid + d, sig_t, sig), np.ones_like(grid)])
        residuals.append(np.linalg.lstsq(fit, target, rcond=None)[1][0])
    return float(shifts[int(np.argmin(residuals))])


# The IMU's x axis and the F/T sensor's z force both follow gravity as the arm
# rotates about the sensor's y axis.
accel, wrench = rows(f"{RUN}_accel"), rows(f"{RUN}_wrench")
latency = yaml.safe_load(declaration("kuka_ft_imu").read_text())["channels"][
    f"{RUN}_accel"
]["latency"]
corrected = delay(
    ds.timestamps[f"{RUN}_accel"],
    accel[:, 0],
    ds.timestamps[f"{RUN}_wrench"],
    wrench[:, 2],
)
print("\nIMU behind the F/T sensor, measured from the data:")
print(
    f"  on the logged clocks       : {(corrected + latency) * 1e3:+5.1f} ms   (documented: +8.4 ms)"
)
print(f"  with the declared latency  : {corrected * 1e3:+5.1f} ms")

# Every robot orientation sample, with the F/T and IMU readings nearest in
# time on the corrected clocks -- within half an IMU period.
view = ds.synchronize(
    reference=f"{RUN}_orientations", method="nearest", tolerance=0.002
)
print(
    f"\n{len(view)} of {len(ds.timestamps[KEYS[0]])} orientation samples have both readings within 2 ms"
)
for key in KEYS[1:]:
    offsets = np.abs(view.time_offsets(key)) * 1e3
    print(
        f"  {key:<24} median offset {np.median(offsets):.2f} ms, max {offsets.max():.2f} ms"
    )
sample = view[len(view) // 2]
print(
    f"\nmid-run: force {np.round(sample.data[KEYS[1]][:3], 2)} N, accel {np.round(sample.data[KEYS[2]], 3)} g"
)
