"""Frame-naming policy: which files the per-frame loader reads/writes.

Covers apairo/core/naming.py -- the reserved-``_`` convention shared by
NPYSLoader (reads), ChannelWriter (writes), and suffixed sub-channel discovery.
"""

import pytest

from apairo.core.naming import (
    frame_stem_is_valid,
    is_frame_file,
    require_frame_stem,
    suffixed_frame_files,
)


def test_frame_stem_is_valid():
    assert frame_stem_is_valid("000000")
    assert not frame_stem_is_valid("000000_intensity")


def test_is_frame_file():
    assert is_frame_file("000000.npy")
    assert not is_frame_file("000000_intensity.npy")
    assert not is_frame_file("000000.txt")


def test_require_frame_stem_rejects_suffix():
    assert require_frame_stem("000000") == "000000"
    with pytest.raises(ValueError):
        require_frame_stem("000000_intensity")
    with pytest.raises(ValueError):
        require_frame_stem("")
    with pytest.raises(ValueError):
        require_frame_stem("a/b")


def test_suffixed_frame_files_filters_and_sorts(tmp_path):
    for stem in ("000002", "000000", "000001"):
        (tmp_path / f"{stem}_intensity.npy").touch()
        (tmp_path / f"{stem}.npy").touch()
    (tmp_path / "000000_range.npy").touch()

    assert suffixed_frame_files(tmp_path, "intensity") == [
        "000000_intensity.npy",
        "000001_intensity.npy",
        "000002_intensity.npy",
    ]


def test_suffixed_frame_files_ignores_ambiguous_stems(tmp_path):
    # A stem with its own '_' before the suffix is not a valid frame stem, so
    # it is excluded even though it ends with the right tail.
    (tmp_path / "000000_foo_intensity.npy").touch()
    (tmp_path / "000001_intensity.npy").touch()

    assert suffixed_frame_files(tmp_path, "intensity") == ["000001_intensity.npy"]
