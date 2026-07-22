"""Export a dataset subset to a new self-contained root.

The write-side dual of loading: :func:`export_dataset` (reached via
:meth:`~apairo.core.abstract_dataset.AbstractDataset.export`) copies a subset of
sequences x channels out of an asynchronous ``RawDataset`` into a fresh root,
regenerating the ``.apairo`` sidecars so the copy is self-contained -- ``apairo
status`` on it reports exactly the exported channels, not the source's.

v1 scope -- **structural subset, asynchronous family only**: whole sequences x
whole channels, a pure file copy (optionally hardlinked) plus regenerated
sidecars. Frame-filtered, re-clocked (``synchronize``) or otherwise transformed
views are rejected: reading samples back through the writers (renumbered stems,
re-encoded bytes) is a different regime -- the "materializing export" extension,
not v1.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from apairo.core.config import (
    CALIBRATION_FILE,
    CONFIG_DIR,
    config_exists,
    read_config,
    write_config,
)


def export_dataset(
    view,
    dest: str | Path,
    *,
    overwrite: bool = False,
    link: bool = False,
) -> Path:
    """Copy *view*'s sequences x channels into *dest* as a new self-contained root.

    See :meth:`~apairo.core.abstract_dataset.AbstractDataset.export` for the
    public contract. *view* is a ``RawDataset`` (root or single sequence),
    optionally narrowed by ``filter_sequences`` and by the channel set passed to
    its constructor.
    """
    from apairo.core.filtered_view import FilteredView
    from apairo.core.root_sequence import RootSequenceMixin
    from apairo.dataset.raw import RawDataset

    # 1. Walk to the concrete on-disk dataset through structural wrappers only.
    #    filter_sequences yields a FilteredView; everything else (synchronize,
    #    cache, window, select, concat) changes the frames or the clock and is
    #    out of the v1 file-copy regime.
    node = view
    while isinstance(node, FilteredView):
        node = node._parent
    # `synchronous` is the class-level async marker (False for the RawDataset
    # family); the `is_synchronous` property is per-instance and reads True on a
    # root, which has no top-level clock -- each sub-sequence carries its own.
    if not isinstance(node, RootSequenceMixin) or getattr(node, "synchronous", True):
        raise ValueError(
            f"export() copies the asynchronous RawDataset family and its "
            f"filter_sequences views only; got {type(view).__name__}. A "
            f"re-clocked (synchronize), cached, windowed, channel-selected or "
            f"concatenated view must be persisted through the materializing "
            f"export path (not in v1). Select channels by constructing "
            f"RawDataset(root, keys=[...])."
        )
    src = node
    root = Path(src.root_dir)
    keys = list(src.keys)
    seq_ids = sorted({str(s) for s in view.frame_sequence_ids})
    if not seq_ids:
        raise ValueError("Nothing to export: the view selects no sequences.")

    # 2. Whole sequences x whole channels only. Rebuild the canonical structural
    #    subset and require the view to match it frame for frame -- any
    #    frame-level filtering fails here rather than silently copying wrong data.
    canonical = RawDataset(root, keys=keys)
    canonical_view = canonical.filter_sequences(seq_ids) if src._is_root else canonical
    if len(canonical_view) != len(view):
        raise ValueError(
            "export() v1 copies whole sequences x whole channels; this view is "
            "frame-filtered. Persist a frame-filtered or transformed view "
            "through the materializing export path (not in v1)."
        )

    # 3. Destination.
    dest = Path(dest).expanduser()
    if dest.exists() and any(dest.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Destination '{dest}' is not empty. Pass overwrite=True to replace it."
            )
        # overwrite REPLACES: clear the destination so stale sequences from a
        # prior export cannot survive and re-enter the regenerated manifest. A
        # genuine incremental 'update' mode belongs in a separate, explicit option.
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    # 4. Copy each selected sequence + regenerate its sidecars.
    src_seq_dirs = (
        {s._sequence_dir.name: s._sequence_dir for s in src.sequences}
        if src._is_root
        else {src._sequence_dir.name: src._sequence_dir}
    )
    selected = set(keys)
    for seq_id in seq_ids:
        _export_sequence(src_seq_dirs[seq_id], dest / seq_id, selected, link=link)

    # 5. Root manifest (always a root output, even for a single sequence).
    name = getattr(src, "name", None) if src._is_root else None
    RawDataset._write_manifest(dest, name=name)
    return dest


def _export_sequence(
    src_seq: Path, dst_seq: Path, selected_public: set[str], *, link: bool
) -> None:
    """Copy the *selected* channels of one sequence and write its channels.yaml."""
    if not config_exists(src_seq):
        raise FileNotFoundError(
            f"Source sequence '{src_seq}' has no .apairo/channels.yaml. "
            f"Run `apairo init` on the source first."
        )
    channels = read_config(src_seq).get("channels", {})
    # A channel is kept when its REAL directory name OR its alias was selected --
    # the request may name either (the alias request language), so matching only
    # the public/alias name would silently drop a channel opened by its real name.
    keep = {
        real: meta
        for real, meta in channels.items()
        if real in selected_public or meta.get("alias", real) in selected_public
    }
    if not keep:
        raise KeyError(
            f"None of the selected channels {sorted(selected_public)} exist in "
            f"'{src_seq.name}' (has: {sorted(channels)})."
        )

    dst_seq.mkdir(parents=True, exist_ok=True)
    # One physical directory may back several channels (a base npys channel and
    # its suffixed sub-channels); copy each unique directory once, whole.
    for directory in sorted(
        {meta.get("directory", real) for real, meta in keep.items()}
    ):
        _copy_tree(src_seq / directory, dst_seq / directory, link=link)

    new_channels: dict = {}
    for real, meta in keep.items():
        entry = dict(meta)
        # Keep the channel self-contained: it carries its own clock (copied with
        # its directory). Drop provenance pointing outside the exported subset so
        # verify_config stays clean; keep it when the referenced channel travels
        # along too.
        _ensure_own_timestamps(src_seq, dst_seq, real, meta, channels, link=link)
        ts_from = entry.get("timestamps_from")
        if ts_from is not None and ts_from not in keep:
            entry.pop("timestamps_from", None)
        if "sources" in entry:
            kept_sources = [s for s in entry["sources"] if s in keep]
            if kept_sources:
                entry["sources"] = kept_sources
            else:
                entry.pop("sources", None)
        new_channels[real] = entry
    write_config(dst_seq, {"version": 1, "channels": new_channels})

    # calibration.yaml is channel-independent -- copied verbatim. Every other
    # sidecar (a third-party metadata.yaml, the source manifest) is dropped: it
    # describes the source, not the subset.
    src_cal = src_seq / CONFIG_DIR / CALIBRATION_FILE
    if src_cal.exists():
        (dst_seq / CONFIG_DIR).mkdir(exist_ok=True)
        shutil.copy2(src_cal, dst_seq / CONFIG_DIR / CALIBRATION_FILE)


def _ensure_own_timestamps(
    src_seq: Path,
    dst_seq: Path,
    real: str,
    meta: dict,
    all_channels: dict,
    *,
    link: bool,
) -> None:
    """Guarantee the exported channel's directory holds its own timestamps.txt.

    A channel that borrows its clock (``timestamps_from``) and has no own file
    -- so its source may be excluded from the subset -- gets the source clock
    materialized beside it, the same normalization ``run_preprocess`` applies."""
    directory = meta.get("directory", real)
    dst_ts = dst_seq / directory / "timestamps.txt"
    if dst_ts.exists():
        return
    ts_from = meta.get("timestamps_from")
    if ts_from and ts_from in all_channels:
        src_dir = all_channels[ts_from].get("directory", ts_from)
        src_ts = src_seq / src_dir / "timestamps.txt"
        if src_ts.exists():
            dst_ts.parent.mkdir(parents=True, exist_ok=True)
            _copy_file(src_ts, dst_ts, link=link)


def _copy_tree(src_dir: Path, dst_dir: Path, *, link: bool) -> None:
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Channel directory not found: {src_dir}")
    copy_function = _link_or_copy if link else shutil.copy2
    shutil.copytree(src_dir, dst_dir, copy_function=copy_function, dirs_exist_ok=True)


def _copy_file(src: Path, dst: Path, *, link: bool) -> None:
    (_link_or_copy if link else shutil.copy2)(str(src), str(dst))


def _link_or_copy(src: str, dst: str) -> None:
    """Hardlink *src* to *dst*; fall back to a copy across filesystems.

    Hardlinks share inodes, so exported frames are near-free and immutable-safe;
    editing a linked file in place would touch the source, but apairo frames are
    write-once."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
