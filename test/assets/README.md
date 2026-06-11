# Real-data test fixtures

Tiny excerpts of real datasets (~300 KB total), used by
`test/dataset/test_smoke_real_data.py` to exercise the full loading path on
data the synthetic tests cannot fake.

| Fixture | Source | Layout | Contents |
|---|---|---|---|
| `mini_rellis` | Rellis-3D | synchronous (ProfiledDataset) | 2 sequences × 5 frames, clouds subsampled to 1024 pts, poses, calib, `.lst` splits |
| `mini_tartan` | TartanDrive v2, `2023-11-14-15-02-21_figure_8` | asynchronous (KITTI layout) | 8 velodyne frames (512 pts), cmd @ ~10 Hz, imu @ ~400 Hz, real timestamps |

Regenerate with `python test/assets/extract_mini_datasets.py` (requires the
full datasets on lab storage — see paths in the script).

Tests must **copy these trees to `tmp_path`** before instantiating a dataset:
apairo writes a `.apairo` sidecar at first load and would dirty the fixtures.
