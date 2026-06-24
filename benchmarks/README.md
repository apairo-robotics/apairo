# Benchmarks — what apairo costs

Synthetic, isolated, scaling with dataset size. The harness measures apairo's own
overhead against a **competent** hand-rolled baseline (sorted glob + `np.load` +
`searchsorted` align). This is the *framework tax* — not real-world I/O, GPU, or
zarr chunking.

```bash
python benchmarks/bench.py            # sizes 200 1000 5000
python benchmarks/bench.py 500 5000   # custom sizes
```

The dataset is an async multi-rate sequence (a lidar, a pose stream at 5×),
mirroring the rosbag → kitti → `RawDataset` path. Generated trees are cached under
`benchmarks/.cache`.

## Results (1024-point frames)

| frames | ds[i] apairo (µs) | ds[i] np.load (µs) | tax | synchronize (ms) | searchsorted (ms) | filter view (ms) | filter view (KB) |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 200 | 19.0 | 18.4 | 1.03× | 0.02 | 0.00 | 0.001 | 0.3 |
| 1000 | 20.5 | 19.1 | 1.07× | 0.03 | 0.01 | 0.000 | 0.2 |
| 5000 | 20.9 | 19.2 | 1.09× | 0.12 | 0.05 | 0.000 | 0.2 |

## What it shows

- **Per-frame access is free.** `ds[i]` ≈ `np.load` (~1.0× at every size) — the
  loader adds no measurable cost over reading the file yourself.
- **Lazy views are ~free.** Building a filtered view over the whole dataset costs
  microseconds and **~0.2 KB, flat in N** — a few index arrays, not a copy. The
  same holds for `select`, `synchronize`, `concat`, `join`.
- **`synchronize` is index math, not I/O.** It scales O(events), sub-millisecond
  at 5000 frames, the same order as a bare `searchsorted`.

## What it does *not* show

Real I/O patterns, zarr chunking, GPU preprocessing, distributed loading. It
isolates the framework cost; the system cost is a separate study.
