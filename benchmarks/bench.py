"""What does apairo cost? Isolated, synthetic, scaling with dataset size.

Run: ``python benchmarks/bench.py`` (sizes optional: ``python benchmarks/bench.py 200 1000 5000``).

It measures apairo's own overhead against a competent hand-rolled baseline, so
the result is the framework tax -- not real-world I/O, GPU, or zarr chunking.
"""
from __future__ import annotations

import statistics
import sys
import time
import tracemalloc

import numpy as np

import apairo as ap
from baseline import NaiveLoader, latest_match
from synth import make_async_sequence

POINTS = 1024
REF = "lidar"


def _ms(fn, repeat: int = 5) -> float:
    """Median milliseconds for one call of *fn*."""
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1e3


def _access_us(loader, indices, repeat: int = 5) -> float:
    """Median microseconds to read one frame, over *indices*."""
    per_call = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for i in indices:
            loader[int(i)]
        per_call.append((time.perf_counter() - t0) / len(indices))
    return statistics.median(per_call) * 1e6


def _peak_kb(fn) -> float:
    """Peak Python memory allocated while running *fn*, in KB."""
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024


def measure(n: int) -> dict:
    root = make_async_sequence(n, POINTS)

    ds = ap.RawDataset(root, keys=["lidar", "pose"])
    lidar = ds.loaders["lidar"]
    naive = NaiveLoader(root / "lidar")
    lidar_ts = np.loadtxt(root / "lidar" / "timestamps.txt")
    pose_ts = np.loadtxt(root / "pose" / "timestamps.txt")
    indices = np.random.default_rng(0).integers(0, len(lidar), size=min(500, len(lidar)))

    sync = ds.synchronize(reference=REF, method="latest")
    keep = np.arange(0, len(sync), 2)

    return {
        "N": len(lidar),
        # 1. per-frame access tax: apairo's loader vs a raw np.load
        "access_ap": _access_us(lidar, indices),
        "access_base": _access_us(naive, indices),
        # 2. synchronize: apairo build vs a searchsorted align
        "sync_ap": _ms(lambda: ds.synchronize(reference=REF, method="latest")),
        "sync_base": _ms(lambda: latest_match(lidar_ts, pose_ts)),
        # 3. a lazy view over the whole dataset: cheap to build, tiny in memory
        "view_ms": _ms(lambda: sync.filter(keep)),
        "view_kb": _peak_kb(lambda: sync.filter(keep)),
    }


_COLUMNS = [
    ("frames", "N", "{:d}"),
    ("ds[i] apairo (µs)", "access_ap", "{:.1f}"),
    ("ds[i] np.load (µs)", "access_base", "{:.1f}"),
    ("tax", None, "{:.2f}x"),
    ("synchronize (ms)", "sync_ap", "{:.2f}"),
    ("searchsorted (ms)", "sync_base", "{:.2f}"),
    ("filter view (ms)", "view_ms", "{:.3f}"),
    ("filter view (KB)", "view_kb", "{:.1f}"),
]


def main(sizes: list[int]) -> None:
    rows = [measure(n) for n in sizes]
    for r in rows:
        r["tax"] = r["access_ap"] / r["access_base"]
    print(" | ".join(h for h, _, _ in _COLUMNS))
    print("-|-".join("-" * len(h) for h, _, _ in _COLUMNS))
    for r in rows:
        print(" | ".join(fmt.format(r[key or head]) for head, key, fmt in _COLUMNS))


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [200, 1000, 5000])
