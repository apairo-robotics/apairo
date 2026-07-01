from __future__ import annotations
from pathlib import Path

from apairo.dataset.async_layout.dataset import _suffix_channel_entries
from apairo.dataset.raw import RawDataset
from apairo.loader import load_profile
from apairo.utils.files import get_files

_PROFILE: dict[str, str] = load_profile(Path(__file__).parent / "profile.yaml")


class TartanKittiDataset(RawDataset):
    r"""TartanDrive v2 -- a :class:`~apairo.dataset.raw.RawDataset` whose channel
    set is the fixed TartanDrive profile.

    Loading, multi-sequence roots, synchronization and preprocessing are exactly
    :class:`RawDataset`'s; this class only pins the canonical TartanDrive
    channels (``profile.yaml``). That means a raw sequence with no ``.apairo``
    bootstraps with the *profile's* loaders rather than ones merely guessed from
    file extensions, and :meth:`describe` can flag profile channels missing on
    disk.

    Single sequence or root directory, auto-detected::

        ds = TartanKittiDataset(seq_dir, keys=["velodyne_0", "cmd"])
        ds = TartanKittiDataset(root_dir, keys=["velodyne_0"])   # all sequences
    """

    available_keys: frozenset = frozenset(_PROFILE)

    def _bootstrap_config(self, sequence_dir: Path) -> dict:
        """Declare the on-disk profile channels, pinning each loader from the
        TartanDrive profile instead of inferring it from file extensions.

        A ``"npys"`` channel may also hold suffixed sub-channel variants (e.g.
        ``velodyne_0/000000_intensity.npy``); those fan out into sibling
        channels (``velodyne_0_intensity``) the same way as an unprofiled
        :class:`~apairo.dataset.raw.RawDataset`.
        """
        channels: dict = {}
        for key in sorted(get_files(str(sequence_dir))):
            if key not in _PROFILE:
                continue
            loader = _PROFILE[key]
            channels[key] = {"loader": loader, "kind": "raw"}
            channel_dir = Path(sequence_dir) / key
            for suffix, frag in _suffix_channel_entries(channel_dir, loader).items():
                channels[f"{key}_{suffix}"] = {"kind": "raw", **frag}
        return {"version": 1, "channels": channels}
