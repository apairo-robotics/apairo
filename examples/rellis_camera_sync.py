"""Load the Rellis-3D camera (+ its sparse image-labels) through apairo's async
family, keying both channels off the timestamp encoded in every filename.

Rellis-3D ships lidar + lidar-labels + poses as a *synchronous* stack (that is the
``Rellis3DDataset`` / ``rellis.yaml`` ``ProfiledDataset``, unchanged). The RGB
camera and the hand image-labels are a *different* alignment problem: the camera is
a dense PTP-locked stream named ``frame<N>-<sec>_<ms>.jpg`` and the image-labels are
a ~half-rate subset named the same way. Neither carries a ``timestamps.txt``, and
the labels are sparse -- so they belong to the asynchronous ``RawDataset`` family,
where each channel derives its own alignment key and ``synchronize()`` aligns them.

Both channels declare ``key: {name: <regex>}``: the key is parsed from the filename
stem *in memory* -- nothing is ever written into the Rellis tree. The regex captures
``<sec>`` and ``<ms>`` and the default combine ``float('.'.join(groups))`` yields the
epoch second (ms is always 3-digit zero-padded, so the decimal join stays monotonic).
The same regex doubles as the enumeration policy, so the ``_`` in every stem -- which
apairo's default frame-file convention reserves for suffixes -- does not block loading.

The Rellis sequence dir is treated as read-only: we stage a scratch sequence that
*symlinks* the two channel directories (no image bytes copied, no mutation of the
source) and write ``channels.yaml`` there.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from apairo.core.config import write_config
from apairo.dataset.raw import RawDataset

# Read-only source: a Rellis-3D sequence. Override the root with APAIRO_RELLIS_ROOT.
# (Note the dual Rellis-3D / Rellis_3D spelling under the vault root -- the camera
# lives in the Rellis-3D tree.)
RELLIS_ROOT = Path(os.environ.get("APAIRO_RELLIS_ROOT", "/mnt/vault-fellowship/rellis"))
RELLIS_SEQ = RELLIS_ROOT / "Rellis-3D" / "00000"

CAMERA = "pylon_camera_node"
LABELS = "pylon_camera_node_label_id"

# frame<N>-<sec>_<ms>  ->  <sec>.<ms>  (epoch seconds). One regex serves as both the
# alignment key and the enumeration policy for these '_'-bearing stems.
TS_KEY = {"name": r"frame\d+-(\d+)_(\d+)"}


def stage_scratch(source_seq: Path, scratch_root: Path) -> Path:
    """A writable sequence dir that symlinks the source channel dirs and carries a
    ``channels.yaml`` -- so the read-only Rellis tree is never touched."""
    seq = scratch_root / source_seq.name
    seq.mkdir(parents=True, exist_ok=True)
    for channel in (CAMERA, LABELS):
        link = seq / channel
        if not link.exists():
            link.symlink_to(source_seq / channel)
    # Both channels: img loader, key parsed from the filename. The label carries the
    # SAME <sec>_<ms> as its camera frame, so keying it by TIMESTAMP lands it exactly
    # on that camera tick. Keying it by frame index instead (frame(\d+)-) would put
    # it in a different number space than the camera clock and align NOTHING.
    write_config(
        seq,
        {
            "version": 1,
            "channels": {
                CAMERA: {"kind": "raw", "loader": "img", "key": TS_KEY},
                LABELS: {"kind": "raw", "loader": "img", "key": TS_KEY},
            },
        },
    )
    return seq


def main() -> None:
    if not (RELLIS_SEQ / CAMERA).is_dir():
        print(
            f"Rellis camera not found at {RELLIS_SEQ} -- set APAIRO_RELLIS_ROOT. Skipping."
        )
        return
    scratch = Path(__file__).parent / "_rellis_scratch"
    seq = stage_scratch(RELLIS_SEQ, scratch)

    ds = RawDataset(seq, keys=[CAMERA, LABELS])

    cam_ts = ds.timestamps[CAMERA]
    dt = np.diff(cam_ts)
    print(f"camera frames : {len(ds.loaders[CAMERA])}")
    print(
        f"camera clock  : monotonic={bool(np.all(dt > 0))}  "
        f"median dt={np.median(dt) * 1000:.1f} ms  t0={cam_ts[0]:.3f}"
    )
    print(
        f"label frames  : {len(ds.loaders[LABELS])} files "
        f"({len(np.unique(ds.timestamps[LABELS]))} distinct timestamps)"
    )

    # Sparse labels onto the dense camera clock: tolerance=0 keeps only the camera
    # ticks that have an exact-timestamp label (the labeled subset); a wider
    # tolerance + method='nearest'/'previous' would attach the closest label.
    view = ds.synchronize(reference=CAMERA, method="nearest", tolerance=0.0)
    print(f"synchronized  : {len(view)} frames, each with {sorted(view[0].data)}")

    assert view.is_synchronous
    assert set(view[0].data) == {CAMERA, LABELS}
    # Zero writes into the Rellis tree: the source has no timestamps.txt after load.
    assert not (RELLIS_SEQ / CAMERA / "timestamps.txt").exists()
    assert not (RELLIS_SEQ / LABELS / "timestamps.txt").exists()
    print("OK: camera dense, labels sparse-synced, nothing written into Rellis.")


if __name__ == "__main__":
    main()

# Verified run (apairo-wt-proto, /home/abresset/dev/apairo/.venv):
#   camera frames : 2847
#   camera clock  : monotonic=True  median dt=100.0 ms  t0=1581624652.750
#   label frames  : 1201 files (1200 distinct timestamps)
#   synchronized  : 1200 frames, each with ['pylon_camera_node', 'pylon_camera_node_label_id']
#   OK: camera dense, labels sparse-synced, nothing written into Rellis.
