"""Intensive-usage soak on synthetic data -- the whole API surface, end to end.

``bench.py`` answers "what does apairo cost"; this answers "does the whole
surface hold together under a realistic, intensive session": bootstrap, reload,
timeline scan, synchronize, preprocess persistence, lazy preview, the chaining
views (filter / select / cache / join / window), shuffled training epochs, and
a pickle roundtrip (multi-worker DataLoader readiness). Every step asserts its
contract -- a non-zero exit means a broken invariant, not a slow one.

Run: ``python benchmarks/soak.py`` (or ``make soak``).
Scale up: ``python benchmarks/soak.py --sequences 8 --frames 500 --points 4096``.
"""

from __future__ import annotations

import argparse
import pickle
import shutil
import tempfile
import time
import tracemalloc
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import apairo as ap
from apairo.core.preprocessor import FramePreprocessor
from apairo.core.sample import Sample

LIDAR_HZ, IMU_HZ, CMD_HZ = 10.0, 100.0, 20.0
EVENTS_PER_LIDAR_FRAME = int(1 + IMU_HZ / LIDAR_HZ + CMD_HZ / LIDAR_HZ)  # 13

_rows: list[tuple[str, float]] = []


@contextmanager
def step(name: str):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    _rows.append((name, dt))
    print(f"  {name:<46} {dt * 1e3:9.1f} ms")


class ZMean(FramePreprocessor):
    """Mean lidar height per frame -- a tiny but real derived channel."""

    output_key = "z_mean"
    output_loader = "npys"
    input_keys = ["lidar"]
    timestamps_from = "lidar"
    sources = ["lidar"]

    def __call__(self, sample: Sample) -> np.ndarray:
        return np.array([sample.data["lidar"][:, 2].mean()], dtype=np.float32)


def _above_ground(points: np.ndarray) -> np.ndarray:
    return points[points[:, 2] > 0.0]


def _stack_z(samples: list[Sample]) -> Sample:
    return Sample(
        data={"z_seq": np.stack([s.data["z_mean"] for s in samples])},
        timestamp=samples[-1].timestamp,
    )


def make_root(root: Path, sequences: int, frames: int, points: int) -> None:
    """A bare multi-rate tree (no .apairo): per-frame lidar (npys, variable N),
    stacked imu and cmd (npy) -- the shape of an extractor output."""
    rng = np.random.default_rng(0)
    duration = frames / LIDAR_HZ
    for s in range(sequences):
        seq = root / f"seq_{s:03d}"
        lidar = seq / "lidar"
        lidar.mkdir(parents=True)
        for i in range(frames):
            n = int(rng.integers(points // 2, points + 1))
            np.save(
                lidar / f"{i:06d}.npy", rng.standard_normal((n, 4)).astype(np.float32)
            )
        np.savetxt(
            lidar / "timestamps.txt",
            np.arange(frames) / LIDAR_HZ + rng.uniform(0, 4e-3, frames),
        )
        for name, rate, width in [("imu", IMU_HZ, 6), ("cmd", CMD_HZ, 2)]:
            d = seq / name
            d.mkdir()
            n = int(duration * rate)
            np.save(
                d / f"{name}.npy", rng.standard_normal((n, width)).astype(np.float32)
            )
            np.savetxt(
                d / "timestamps.txt",
                np.arange(n) / rate + rng.uniform(0, 0.4 / rate, n),
            )


def soak(root: Path, sequences: int, frames: int, points: int, epochs: int) -> None:
    n_events = sequences * frames * EVENTS_PER_LIDAR_FRAME

    with step("generate synthetic tree"):
        make_root(root, sequences, frames, points)

    with step("bootstrap load (bare tree, no .apairo)"):
        ds = ap.RawDataset(root)
    assert set(ds.keys) == {"lidar", "imu", "cmd"}, ds.keys
    assert len(ds) == n_events, (len(ds), n_events)

    with step("reload from the written sidecars"):
        ds = ap.RawDataset(root, keys=["lidar", "imu", "cmd"])
    assert len(ds) == n_events

    with step("full timeline scan (event order + provenance)"):
        last: dict = {}
        for i in range(len(ds)):
            sample = ds[i]
            info = ds.frame_info(i)
            assert len(sample.data) == 1
            assert info.channel in sample.data
            assert sample.timestamp >= last.get(info.sequence, -np.inf)
            last[info.sequence] = sample.timestamp
    assert len(last) == sequences

    with step("materialize derived channel (run_preprocess)"):
        ds.run_preprocess(ZMean())

    with step("reload derived at root and sequence level"):
        assert "z_mean" in ap.RawDataset(root).keys
        seq_derived = ap.RawDataset(root / "seq_000", keys=["z_mean"])
        seq_lidar = ap.RawDataset(root / "seq_000", keys=["lidar"])
    assert len(seq_derived) == frames
    for i in range(0, frames, max(1, frames // 7)):
        np.testing.assert_allclose(
            seq_derived[i].data["z_mean"],
            seq_lidar[i].data["lidar"][:, 2].mean(),
            rtol=1e-6,
        )

    with step("lazy preview == materialized channel"):
        preview = ap.RawDataset(root / "seq_000", keys=["lidar"]).transform(ZMean())
        for i in range(0, frames, max(1, frames // 7)):
            np.testing.assert_allclose(
                preview[i].data["z_mean"], seq_derived[i].data["z_mean"], rtol=1e-6
            )

    with step("synchronize root (previous, tol 60 ms)"):
        ds_all = ap.RawDataset(root)
        sync = ds_all.synchronize(reference="lidar", method="previous", tolerance=0.06)
        n_sync = len(sync)
    assert sync.is_synchronous
    # At most the first lidar tick per sequence lacks a previous imu/cmd event.
    assert sequences * (frames - 1) <= n_sync <= sequences * frames, n_sync
    probe = sync[n_sync // 2]
    assert set(probe.data) == {"lidar", "imu", "cmd", "z_mean"}
    assert probe.timestamp is not None

    with step("chain: select -> cache -> join -> filter -> transform"):
        prior = sync.select(["z_mean"]).cache()
        even = np.arange(0, n_sync, 2)
        train = (
            sync.select(["lidar", "imu"])
            .join(prior)
            .filter(even)
            .transform("lidar", _above_ground)
        )
    assert len(train) == len(even)
    sample = train[1]
    assert set(sample.data) == {"lidar", "imu", "z_mean"}
    assert (sample.data["lidar"][:, 2] > 0.0).all()

    with step(f"{epochs} shuffled epochs over the pipeline"):
        tracemalloc.start()
        rng = np.random.default_rng(1)
        probe_idx = len(train) // 3
        probes = []
        for _ in range(epochs):
            order = rng.permutation(len(train))
            for i in order:
                s = train[int(i)]
                assert s.data["lidar"].ndim == 2
            probes.append(train[probe_idx].data["z_mean"].copy())
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    for p in probes[1:]:  # deterministic pipeline -> identical across epochs
        np.testing.assert_array_equal(p, probes[0])
    print(f"  {'  peak python memory during epochs':<46} {peak / 1e6:9.1f} MB")

    with step("window (size 3, stacked z-mean, drop boundary)"):
        seq_sync = ap.RawDataset(root / "seq_000").synchronize(reference="lidar")
        win = seq_sync.select(["z_mean"]).window(3, reduce=_stack_z, boundary="drop")
        w = win[len(win) // 2]
    assert w.data["z_seq"].shape == (3, 1)

    with step("concat along the frame axis"):
        doubled = sync.concat(sync)
    assert len(doubled) == 2 * n_sync

    with step("pickle roundtrip (multi-worker readiness)"):
        clone = pickle.loads(pickle.dumps(train))
    assert len(clone) == len(train)
    np.testing.assert_allclose(
        clone[probe_idx].data["z_mean"], train[probe_idx].data["z_mean"]
    )

    total = sum(dt for _, dt in _rows)
    print(
        f"\nSOAK PASSED -- {len(_rows)} steps, {n_events} events, "
        f"{n_sync} synced frames, {total:.1f} s total"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sequences", type=int, default=4)
    parser.add_argument(
        "--frames", type=int, default=120, help="lidar frames per sequence"
    )
    parser.add_argument(
        "--points", type=int, default=1024, help="max points per lidar frame"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="work directory (default: a fresh temp dir, deleted on success)",
    )
    args = parser.parse_args()

    keep = args.root is not None
    root = args.root or Path(tempfile.mkdtemp(prefix="apairo_soak_")) / "soak_ds"
    print(
        f"soak: {args.sequences} sequences x {args.frames} lidar frames, "
        f"<= {args.points} pts/frame, {args.epochs} epochs -> {root}"
    )
    soak(root, args.sequences, args.frames, args.points, args.epochs)
    if not keep:
        shutil.rmtree(root.parent, ignore_errors=True)


if __name__ == "__main__":
    main()
