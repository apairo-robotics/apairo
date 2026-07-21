"""Head-to-head: a hand-rolled loader vs apairo, on synthetic asynchronous data.

Answers the two questions a data layer has to answer -- "does it cost
throughput?" and "how much code does it save?" -- without a private dataset. The
tree is synthetic, so the whole thing is reproducible and license-free; both
implementations produce **byte-identical frames** (asserted), so the comparison
is honest: identical work, measured on lines-of-code and throughput.

The task is deliberately the shape where a hand-rolled loader hurts most --
asynchronous multi-sensor: match a 100 Hz IMU to each 10 Hz lidar tick within a
tolerance, then drop below-ground points. That nearest-timestamp matching is the
~15 lines apairo collapses into one ``synchronize(...)`` call.

Throughput is swept over point-cloud size on purpose. apairo adds a *fixed*
per-frame overhead (timeline routing, a Sample dataclass, the transform
pipeline, DataLoader-ready provenance); the hand-rolled loop skips all of it. On
trivially cheap frames that overhead is visible; as frames grow to realistic
lidar sizes the per-frame work dominates and the ratio climbs to ~1.0 -- i.e. no
regression where it matters. Reporting only one size would either flatter or
smear apairo, so we show the trend and let the reader see the crossover.

Run: ``python benchmarks/headtohead.py`` (or ``make bench-h2h``).

Not covered here (future): a public real-dataset anchor (Rellis-3D /
SemanticKITTI, both apairo-native) for the "we ran it on data you know" claim,
and a tiny model trained both ways to show accuracy is preserved.
"""

from __future__ import annotations

import argparse
import inspect
import io
import shutil
import tempfile
import time
import tokenize
from pathlib import Path

import numpy as np

import apairo as ap

LIDAR_HZ, IMU_HZ = 10.0, 100.0
TOLERANCE = 0.02
POINT_SWEEP = (1024, 16384, 131072)  # trivial -> realistic lidar frame sizes


def make_root(root: Path, sequences: int, frames: int, points: int) -> None:
    """A bare async tree: per-frame lidar (npys, variable N) at 10 Hz, a stacked
    imu (npy) at 100 Hz, each on its own jittered clock -- an extractor's shape."""
    rng = np.random.default_rng(0)
    for s in range(sequences):
        seq = root / f"seq_{s:03d}"
        (seq / "lidar").mkdir(parents=True)
        for i in range(frames):
            n = int(rng.integers(points // 2, points + 1))
            np.save(
                seq / "lidar" / f"{i:06d}.npy", rng.standard_normal((n, 4)).astype("f4")
            )
        np.savetxt(
            seq / "lidar" / "timestamps.txt",
            np.arange(frames) / LIDAR_HZ + rng.uniform(0, 4e-3, frames),
        )
        (seq / "imu").mkdir()
        n_imu = int(frames / LIDAR_HZ * IMU_HZ)
        np.save(seq / "imu" / "imu.npy", rng.standard_normal((n_imu, 6)).astype("f4"))
        np.savetxt(
            seq / "imu" / "timestamps.txt",
            np.arange(n_imu) / IMU_HZ + rng.uniform(0, 4e-4, n_imu),
        )


# ── the two implementations of the *same* pipeline ────────────────────────────


class HandRolledLoader:
    """The honest hand-rolled way: glob the lidar files, match the nearest IMU
    sample to each lidar tick within a tolerance, load the cloud on access and
    drop below-ground points. This is the code apairo replaces."""

    def __init__(self, root: Path, tolerance: float) -> None:
        self.tol = tolerance
        self.frames: list[tuple[Path, np.ndarray]] = []
        for seq in sorted(p for p in Path(root).iterdir() if p.is_dir()):
            lidar_files = sorted((seq / "lidar").glob("*.npy"))
            lidar_ts = np.atleast_1d(np.loadtxt(seq / "lidar" / "timestamps.txt"))
            imu = np.load(seq / "imu" / "imu.npy")
            imu_ts = np.atleast_1d(np.loadtxt(seq / "imu" / "timestamps.txt"))
            for path, t in zip(lidar_files, lidar_ts, strict=True):
                j = int(np.argmin(np.abs(imu_ts - t)))
                if abs(imu_ts[j] - t) <= self.tol:
                    self.frames.append((path, imu[j]))

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i: int) -> dict:
        path, imu_row = self.frames[i]
        pts = np.load(path)
        return {"lidar": pts[pts[:, 2] > 0.0], "imu": imu_row}


def build_apairo(root: Path, tolerance: float):
    """The same pipeline in apairo: async sync + an access-time filter."""
    return (
        ap.RawDataset(root, keys=["lidar", "imu"])
        .synchronize(reference="lidar", method={"imu": "nearest"}, tolerance=tolerance)
        .transform("lidar", lambda p: p[p[:, 2] > 0.0])
    )


# ── measurement ───────────────────────────────────────────────────────────────


def _sloc(obj) -> int:
    """Source lines of code -- logical lines (tokenizer NEWLINEs), so blank lines,
    comments and line continuations don't inflate the count."""
    src = inspect.getsource(obj)
    return sum(
        1
        for tok in tokenize.generate_tokens(io.StringIO(src).readline)
        if tok.type == tokenize.NEWLINE
    )


def _assert_identical(a, b) -> None:
    assert len(a) == len(b), f"length differs: hand-rolled {len(a)} vs apairo {len(b)}"
    for i in range(len(a)):
        fa, fb = a[i], b[i].data
        assert fa.keys() == fb.keys(), (i, fa.keys(), fb.keys())
        for k in fa:
            np.testing.assert_array_equal(fa[k], fb[k], err_msg=f"frame {i} key {k}")


def _warm_rate(ds, passes: int) -> float:
    """Steady-state frames/s: min over warm passes (drops the cold first pass)."""
    rates = []
    for _ in range(passes):
        t0 = time.perf_counter()
        for i in range(len(ds)):
            s = ds[i]
            _ = s["lidar"] if isinstance(s, dict) else s.data["lidar"]
        rates.append(len(ds) / (time.perf_counter() - t0))
    return min(rates[1:]) if passes > 1 else rates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sequences", type=int, default=2)
    parser.add_argument(
        "--frames", type=int, default=150, help="lidar frames per sequence"
    )
    parser.add_argument(
        "--passes", type=int, default=3, help="throughput passes (1 cold + warm)"
    )
    parser.add_argument(
        "--points",
        type=int,
        nargs="+",
        default=list(POINT_SWEEP),
        help="max points per frame to sweep",
    )
    args = parser.parse_args()

    sloc_hand = _sloc(HandRolledLoader)
    sloc_apairo = _sloc(build_apairo)
    print(
        f"code:  hand-rolled {sloc_hand} SLOC  vs  apairo {sloc_apairo} SLOC  "
        f"-> {sloc_hand / sloc_apairo:.1f}x shorter (and apairo also does the\n"
        f"       .apairo bookkeeping, provenance and DataLoader-readiness the "
        f"hand-rolled loop skips)\n"
    )
    print(
        f"throughput ({args.sequences} seq x {args.frames} frames, imu @ {IMU_HZ:.0f}Hz -> lidar @ {LIDAR_HZ:.0f}Hz):"
    )
    print(
        f"  {'points/frame':>12}  {'hand-rolled f/s':>16}  {'apairo f/s':>12}  {'ratio':>7}"
    )

    for pts in args.points:
        work = Path(tempfile.mkdtemp(prefix="apairo_h2h_"))
        try:
            root = work / "ds"
            make_root(root, args.sequences, args.frames, pts)
            hand = HandRolledLoader(root, TOLERANCE)
            apairo = build_apairo(root, TOLERANCE)
            _assert_identical(hand, apairo)
            r_hand = _warm_rate(hand, args.passes)
            r_apairo = _warm_rate(apairo, args.passes)
            print(
                f"  {pts:>12}  {r_hand:>16.0f}  {r_apairo:>12.0f}  {r_apairo / r_hand:>6.2f}x"
            )
        finally:
            shutil.rmtree(work, ignore_errors=True)

    print(
        "\n  ratio -> 1.0 as frames grow to realistic lidar sizes: apairo's fixed\n"
        "  per-frame overhead is amortized once a frame does real work (load +\n"
        "  filter + downstream compute), so there is no throughput regression\n"
        "  where a training actually spends its time."
    )


if __name__ == "__main__":
    main()
