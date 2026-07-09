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
from pathlib import Path

import numpy as np
from baseline import NaiveLoader, latest_match
from synth import make_async_sequence

import apairo as ap

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


def _access_tax(root, points: int) -> tuple[float, float]:
    """(apairo µs, np.load µs) to read one lidar frame of *points* points."""
    ds = ap.RawDataset(root, keys=["lidar"])
    lidar, naive = ds.loaders["lidar"], NaiveLoader(root / "lidar")
    idx = np.random.default_rng(0).integers(0, len(lidar), size=min(500, len(lidar)))
    return _access_us(lidar, idx), _access_us(naive, idx)


def measure(n: int) -> dict:
    """Cost vs dataset size N (frames), at a fixed frame size."""
    root = make_async_sequence(n, POINTS)
    ds = ap.RawDataset(root, keys=["lidar", "pose"])
    lidar_ts = np.loadtxt(root / "lidar" / "timestamps.txt")
    pose_ts = np.loadtxt(root / "pose" / "timestamps.txt")

    sync = ds.synchronize(reference=REF, method="previous")
    keep = np.arange(0, len(sync), 2)
    access_ap, access_base = _access_tax(root, POINTS)

    return {
        "N": len(ds.loaders["lidar"]),
        "access_ap": access_ap,
        "access_base": access_base,
        # synchronize: apairo build vs a bare searchsorted align
        "sync_ap": _ms(lambda: ds.synchronize(reference=REF, method="previous")),
        "sync_base": _ms(lambda: latest_match(lidar_ts, pose_ts)),
        # a lazy view over the whole dataset: cheap to build, tiny in memory
        "view_ms": _ms(lambda: sync.filter(keep)),
        "view_kb": _peak_kb(lambda: sync.filter(keep)),
    }


def measure_frame_size(points: int, n: int = 300) -> dict:
    """Access tax vs frame size -- the loader's fixed cost fades as frames grow."""
    root = make_async_sequence(n, points)
    access_ap, access_base = _access_tax(root, points)
    return {
        "points": points,
        "access_ap": access_ap,
        "access_base": access_base,
        "tax": access_ap / access_base,
    }


class _Norm(ap.FramePreprocessor):
    """A small per-frame feature, to time compute-once vs reload."""

    output_key = "feat"
    output_loader = "npys"
    input_keys = ["lidar"]
    timestamps_from = "lidar"

    def __call__(self, sample):
        return np.linalg.norm(sample.data["lidar"][:, :3], axis=1).astype(np.float32)


def measure_preprocess(n: int = 1000) -> dict:
    """Preprocess is run-once: the first pass computes + writes, later runs read."""
    root = make_async_sequence(n, POINTS)

    def reload():
        feat = ap.RawDataset(root, keys=["feat"]).loaders["feat"]
        for i in range(len(feat)):
            feat[i]

    compute = _ms(lambda: ap.RawDataset.run_preprocess(_Norm(), root, overwrite=True))
    return {"N": n, "compute_ms": compute, "reload_ms": _ms(reload)}


def _table(rows: list[dict], columns: list[tuple]) -> None:
    print(" | ".join(h for h, _, _ in columns))
    print("-|-".join("-" * len(h) for h, _, _ in columns))
    for r in rows:
        print(" | ".join(fmt.format(r[k]) for _, k, fmt in columns))


_SCALING = [
    ("frames", "N", "{:d}"),
    ("ds[i] apairo (µs)", "access_ap", "{:.1f}"),
    ("ds[i] np.load (µs)", "access_base", "{:.1f}"),
    ("tax", "tax", "{:.2f}x"),
    ("synchronize (ms)", "sync_ap", "{:.2f}"),
    ("searchsorted (ms)", "sync_base", "{:.2f}"),
    ("filter view (ms)", "view_ms", "{:.3f}"),
    ("filter view (KB)", "view_kb", "{:.1f}"),
]
_FRAME_SIZE = [
    ("points/frame", "points", "{:d}"),
    ("ds[i] apairo (µs)", "access_ap", "{:.1f}"),
    ("ds[i] np.load (µs)", "access_base", "{:.1f}"),
    ("tax", "tax", "{:.2f}x"),
]
_PREPROCESS = [
    ("frames", "N", "{:d}"),
    ("compute once (ms)", "compute_ms", "{:.1f}"),
    ("reload (ms)", "reload_ms", "{:.1f}"),
]


def _plot(scaling: list[dict], frame_size: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed -- pip install apairo[bench] for plots)")
        return
    out = Path(__file__).parent

    # A lazy view costs a few KB while the data it views grows without bound.
    N = [r["N"] for r in scaling]
    data_kb = [n * POINTS * 16 / 1024 for n in N]  # lidar bytes (points x 4 x 4)
    fig, ax = plt.subplots()
    ax.plot(N, data_kb, "o-", label="dataset on disk")
    ax.plot(N, [r["view_kb"] for r in scaling], "o-", label="filter view in RAM")
    ax.set(
        yscale="log",
        xlabel="frames",
        ylabel="KB (log)",
        title="A lazy view is a few KB, whatever the data size",
    )
    ax.legend()
    fig.savefig(out / "view_memory.png", dpi=120, bbox_inches="tight")

    # Per-frame access: apairo's loader sits right on top of np.load.
    pts = [r["points"] for r in frame_size]
    fig, ax = plt.subplots()
    ax.plot(pts, [r["access_base"] for r in frame_size], "o-", label="np.load")
    ax.plot(pts, [r["access_ap"] for r in frame_size], "o--", label="apairo ds[i]")
    ax.set(
        xscale="log",
        xlabel="points / frame",
        ylabel="µs / frame",
        title="Per-frame access: apairo tracks np.load",
    )
    ax.legend()
    fig.savefig(out / "access.png", dpi=120, bbox_inches="tight")
    print("\nwrote view_memory.png, access.png")


def main(sizes: list[int], plot: bool = False) -> None:
    print("## Cost vs dataset size (frames)\n")
    scaling = [measure(n) for n in sizes]
    for r in scaling:
        r["tax"] = r["access_ap"] / r["access_base"]
    _table(scaling, _SCALING)

    print("\n## Access tax vs frame size\n")
    frame_size = [measure_frame_size(p) for p in (256, 1024, 4096, 16384)]
    _table(frame_size, _FRAME_SIZE)

    print("\n## Preprocess: compute once, reload free\n")
    _table([measure_preprocess()], _PREPROCESS)

    if plot:
        _plot(scaling, frame_size)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--plot"]
    main([int(a) for a in args] or [200, 1000, 5000], plot="--plot" in sys.argv)
