"""Persist an expensive *sequential* derived channel off-dataset, reload free.

The on-disk sibling of ``join_cached_prior.py``. Same shape — compute a heavy
prior once, ``join`` it onto the live dataset — but here the prior is:

  * **sequential/stateful**: each frame depends on the ones before it (a rolling
    accumulation, an odometry replay, a running voxel map). ``transform`` can't
    hold that state safely (it breaks under a shuffling DataLoader); this is
    exactly what a sequential pass computes.
  * **persisted on disk, in a project-local directory** — not in RAM (``cache``
    dies with the process, so a 40-min replay is paid again every run) and not
    written back into the dataset (it may be shared/read-only, and the config
    churns while you experiment).

apairo has no single "materialise this preprocessor over there" call, but the
three pieces compose into one cleanly:

  1. ``ChannelWriter(cache_dir, ...)`` — writes ``cache_dir/<channel>/*.npy`` +
     ``timestamps.txt`` + ``cache_dir/.apairo/channels.yaml``. The dir need not
     exist and is never the shared dataset.
  2. ``RawDataset(cache_dir)`` — a lone ``kind: preprocess`` channel loads as a
     standalone single-channel dataset (``.apairo`` bootstraps it).
  3. ``base.join(channel_ds)`` — per-index channel merge. ``join`` requires
     equal length (so align to the sensor clock first with ``synchronize``) and
     no key collision (default ``on_collision="raise"``).

Config churn is a directory per config (``slidevox_v015`` vs ``slidevox_v020``);
throwing a config away is ``rm -rf`` on its dir. The shared dataset is never
touched, and ``.apairo`` stays the source of truth in each root it lives in.

Run it twice: the first run computes, the second reloads free.
"""

import os
from pathlib import Path

import numpy as np

from apairo import ChannelWriter, RawDataset

# Runnable out of the box against the bundled test asset; override for real data.
ROOT = Path(
    os.environ.get(
        "APAIRO_ROOT",
        Path(__file__).resolve().parents[1] / "test/assets/mini_tartan/figure_8",
    )
)
REFERENCE = "velodyne_0"  # the clock the derived channel is 1:1 with
CHANNEL = "slidevox_feats"

# --- config that churns; its value names the cache dir -----------------------
VOX = 0.15
CACHE_DIR = (
    Path(os.environ.get("APAIRO_CACHE_ROOT", Path(__file__).parent / "_cache"))
    / f"slidevox_v{int(VOX * 100):03d}"
)


def replay_sequential(sync):
    """Stateful sequential pass — a stand-in for a slidevox / odometry replay.

    Yields one ``(feature, timestamp)`` per reference frame. The feature carries
    running state (here a rolling mean of z), so frame *i* depends on 0..i — the
    property that rules out a plain ``transform`` and demands a single ordered
    pass. Swap the body for your real accumulation.
    """
    running_z = 0.0
    for i in range(len(sync)):
        s = sync[i]
        pts = s.data[REFERENCE]
        running_z = 0.9 * running_z + 0.1 * float(pts[:, 2].mean())
        feat = np.array([len(pts), running_z], dtype=np.float32)
        yield feat, s.timestamp


def materialise_once(sync):
    """Compute + persist the channel, or skip if the cache dir already holds it.

    This is apairo's 'sweep once, reload free' applied to a derived channel:
    the expensive pass runs only when the cache is cold.
    """
    if (CACHE_DIR / CHANNEL).is_dir():
        print(f"reusing cached channel at {CACHE_DIR}")
        return
    print(f"computing {CHANNEL} (cold cache) → {CACHE_DIR}")
    with ChannelWriter(
        CACHE_DIR,
        CHANNEL,
        loader="npys",
        timestamps_from=REFERENCE,
        sources=[REFERENCE],
    ) as w:
        for i, (feat, ts) in enumerate(replay_sequential(sync)):
            w.add(feat, stem=f"{i:06d}", timestamp=ts)


# ---------------------------------------------------------------------------
# 1. Base dataset, aligned to the reference clock so it is 1:1 with the channel
# ---------------------------------------------------------------------------

base = RawDataset(ROOT, keys=[REFERENCE])
sync = base.synchronize(reference=REFERENCE)  # length == number of ref frames
print(f"base: {len(sync)} frames on the '{REFERENCE}' clock")

# ---------------------------------------------------------------------------
# 2. Materialise the sequential prior into a project-local cache dir (once)
# ---------------------------------------------------------------------------

materialise_once(sync)

# ---------------------------------------------------------------------------
# 3. Reload the cache dir as a standalone dataset and join it onto the base
# ---------------------------------------------------------------------------

feats = RawDataset(CACHE_DIR)
print(f"cached channel: {feats.keys}  ({len(feats)} frames)")

combined = sync.join(feats)  # per-index merge; raises on length/key mismatch

sample = combined[0]
print(f"combined keys : {combined.keys}")
print(f"{REFERENCE:14}: {sample.data[REFERENCE].shape}")
print(f"{CHANNEL:14}: {sample.data[CHANNEL]}")

# Throw a config away with `rm -rf` on its dir; the shared dataset under ROOT is
# never written. To sweep VOX values, loop over configs — each gets its own dir,
# computed once, reloaded free thereafter.
