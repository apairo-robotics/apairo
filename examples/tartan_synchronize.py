"""Resample an asynchronous TartanDrive sequence into synchronous frames.

Each sensor fires at its own rate; synchronize() aligns them on a reference
clock so every sample holds all channels — ready for a map-style DataLoader.
"""

from apairo import TartanKittiDataset

SEQ_DIR = "/data/tartan/2024-01-01_forest"

ds = TartanKittiDataset(SEQ_DIR, keys=["velodyne_0", "image_left", "cmd"])
print("Timeline events     :", len(ds))

ds_sync = ds.synchronize(reference="velodyne_0", method="latest", tolerance=0.1)
print("Synchronized frames :", len(ds_sync))
print("Reference channel   :", ds_sync.reference)

sample = ds_sync[0]
print("Channels per frame  :", sorted(sample.data.keys()))
print("Frame timestamp     :", sample.timestamp)

# How stale is each channel relative to the reference clock?
for key in ds.keys:
    offsets = ds_sync.time_offsets(key)
    print(f"{key:<12} mean |dt| = {abs(offsets).mean() * 1000:.1f} ms")

# The view is synchronous: filtering, transforms and shuffling all work.
ds_train = ds_sync.filter("velodyne_0", lambda pts: pts.shape[0] > 1000)
print("Frames after filter :", len(ds_train))
