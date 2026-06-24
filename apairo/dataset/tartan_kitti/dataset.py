from __future__ import annotations
from pathlib import Path

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
        TartanDrive profile instead of inferring it from file extensions."""
        return {
            "version": 1,
            "channels": {
                key: {"loader": _PROFILE[key], "kind": "raw"}
                for key in sorted(get_files(str(sequence_dir)))
                if key in _PROFILE
            },
        }
