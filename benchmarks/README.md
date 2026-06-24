# Benchmarks — what apairo costs

Synthetic, isolated, scaling with dataset size. The harness measures apairo's own
overhead against a **competent** hand-rolled baseline (sorted glob + `np.load` +
`searchsorted` align). This is the *framework tax* — not real-world I/O, GPU, or
zarr chunking.

```bash
python benchmarks/bench.py                 # sizes 200 1000 5000
python benchmarks/bench.py 500 5000 --plot # custom sizes + write the plots
```

The dataset is an async multi-rate sequence (a lidar, a pose stream at 5×),
mirroring the rosbag → kitti → `RawDataset` path. Generated trees are cached under
`benchmarks/.cache`. Plots need `pip install apairo[bench]`.

## Cost vs dataset size (1024-point frames)

| frames | ds[i] apairo (µs) | ds[i] np.load (µs) | tax | synchronize (ms) | searchsorted (ms) | filter view (ms) | filter view (KB) |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 200 | 19.6 | 19.1 | 1.03× | 0.02 | 0.00 | 0.001 | 0.3 |
| 1000 | 19.9 | 19.8 | 1.00× | 0.03 | 0.01 | 0.000 | 0.2 |
| 5000 | 20.9 | 19.2 | 1.09× | 0.12 | 0.05 | 0.000 | 0.2 |

`--plot` writes `view_memory.png`: the filter view stays a few KB while the data
it views climbs into the tens of MB.

## Access tax vs frame size (300 frames)

| points/frame | ds[i] apairo (µs) | ds[i] np.load (µs) | tax |
|--:|--:|--:|--:|
| 256 | 18.9 | 18.3 | 1.04× |
| 1024 | 19.5 | 18.9 | 1.03× |
| 4096 | 23.3 | 21.3 | 1.09× |
| 16384 | 35.2 | 34.8 | 1.01× |

`--plot` writes `access.png`: `apairo ds[i]` sits right on top of `np.load`, both
rising with the frame size.

## Preprocess: compute once, reload free

| frames | compute once (ms) | reload (ms) |
|--:|--:|--:|
| 1000 | ~110 | ~20 |

## What it shows

- **Per-frame access is free.** `ds[i]` sits on top of `np.load` (~1.0× at every
  size); at large frames the data dominates and the two are indistinguishable.
- **Lazy views are ~free.** A filtered view over the whole dataset is ~0.2 KB,
  flat in N, while the data it views grows without bound — a few index arrays,
  not a copy. The same holds for `select`, `synchronize`, `concat`, `join`.
- **`synchronize` is index math, not I/O.** O(events), sub-millisecond at 5000
  frames, the same order as a bare `searchsorted`.
- **Preprocess amortizes.** Computing + persisting a channel costs ~5× a reload;
  every epoch after the first is just a read.

## What it does *not* show

Real I/O patterns, zarr chunking, GPU preprocessing, distributed loading. It
isolates the framework cost; the system cost is a separate study.
