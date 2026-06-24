"""Load a TartanDrive v2 sequence and iterate over its channels."""

import os

from apairo.dataset import TartanKittiDataset

SEQ_DIR = os.environ.get("APAIRO_TARTAN_SEQ", "/data/tartan/2024-01-01_forest")

# On first run, .apairo is created automatically from the discovered raw channels.
# Pass the channels you want; an async sequence holds one event per timeline index.
ds = TartanKittiDataset(SEQ_DIR, keys=["velodyne_0", "cmd", "multisense_imu"])
print("Channels        :", ds.keys)
print("Timeline length :", len(ds))

for sample in ds:
    key = list(sample.data.keys())[0]
    print(f"t={sample.timestamp:.3f}  {key}: {sample.data[key].shape}")
