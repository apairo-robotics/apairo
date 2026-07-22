"""Regressions from the synchronous-clock + array_file features (see the audit).

Each test is a bug the audit found: ConcatDataset misreading a clocked
synchronous dataset, stacked loaders handing out views into their cache, and the
profile clock crashing on anchorless / partially-covered loads.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from apairo.core.profiled_dataset import ProfiledDataset
from apairo.dataset import Rellis3DDataset
from apairo.loader.npy_loader import NPYLoader


class _KittiDS(ProfiledDataset):
    _profile = "semantic_kitti.yaml"


MINI = Path(__file__).parents[1] / "assets" / "mini_rellis"


@pytest.fixture
def rellis(tmp_path):
    dst = tmp_path / "rellis"
    shutil.copytree(MINI, dst)
    return str(dst)


# -- H1: ConcatDataset + the per-frame clock ------------------------------------


def test_concat_of_clocked_synchronous_datasets(rellis):
    a = Rellis3DDataset(rellis, keys=["lidar", "labels"])
    assert a.is_synchronous and a.timestamps is not None  # camera clock present
    c = a.concat(Rellis3DDataset(rellis, keys=["lidar", "labels"]))
    assert c.is_synchronous is True  # was False before the fix
    np.testing.assert_allclose(
        c.timestamps, np.concatenate([a.timestamps, a.timestamps])
    )
    assert len(c) == 2 * len(a)


def test_concat_of_views_does_not_crash(rellis):
    # concat reads child clocks via the is_synchronous protocol, not a bare
    # attribute the view wrappers don't define.
    v = Rellis3DDataset(rellis, keys=["lidar", "labels"]).select(["lidar"]).cache()
    c = v.concat(v)
    assert c.is_synchronous is True
    assert c.timestamps is None or len(c.timestamps) == len(c)


# -- H2: stacked loaders must return copies, not views into the cache -----------


def test_npy_loader_returns_a_copy(tmp_path):
    np.save(tmp_path / "poses.npy", np.zeros((3, 4)))
    ld = NPYLoader(tmp_path, file="poses.npy")
    ld[0][:] += 5.0  # mutate the returned row in place
    np.testing.assert_array_equal(ld[0], np.zeros(4))  # cache uncorrupted


def test_stacked_poses_returns_a_copy(rellis):
    ds = Rellis3DDataset(rellis, keys=["lidar", "poses"])
    before = ds[0].data["poses"].copy()
    ds[0].data["poses"] += 100.0
    np.testing.assert_array_equal(ds[0].data["poses"], before)  # unchanged on re-read


# -- H4/H5: clock crashes become graceful clockless -----------------------------


def test_anchorless_load_with_clock_source_present_is_clockless(rellis):
    ds = Rellis3DDataset(rellis, keys=["poses"])  # no per-frame anchor
    assert ds.timestamps is None
    assert ds[0].timestamp is None


def test_partial_clock_coverage_is_clockless(rellis):
    shutil.rmtree(Path(rellis) / "Rellis-3D" / "00001" / "pylon_camera_node")
    ds = Rellis3DDataset(rellis, keys=["lidar", "labels"])  # 00001 lacks the camera
    assert ds.timestamps is None
    assert len(ds) == 10


# -- H3: non-zero-padded frames must sort numerically, not lexicographically ----


def test_non_zero_padded_frames_align_with_stacked_clock(tmp_path):
    # Frames named 0..11 (not zero-padded). A lexicographic sort orders them
    # 0,1,10,11,2,... so frame '10' lands at position 2 and would be assigned
    # times.txt line 2 (and any stacked/sidecar row 2) -- a silent misalignment.
    seq = tmp_path / "sequences" / "00"
    (seq / "velodyne").mkdir(parents=True)
    (seq / "labels").mkdir(parents=True)
    n = 12
    for i in range(n):
        # encode the frame index into the cloud content to verify frame order
        np.full((4, 4), float(i), np.float32).tofile(seq / "velodyne" / f"{i}.bin")
        np.zeros(4, np.int32).tofile(seq / "labels" / f"{i}.label")
    (seq / "times.txt").write_text("\n".join(f"{i / 10:.1f}" for i in range(n)))

    ds = _KittiDS(str(tmp_path), keys=["lidar", "labels"])
    for k in range(n):
        assert ds[k].data["lidar"][0, 0] == pytest.approx(float(k))  # frame k, in order
        assert ds[k].timestamp == pytest.approx(k / 10)  # clock line k, aligned


# -- H10: cache() must keep the parent's sequence provenance --------------------


def test_cache_preserves_sequence_provenance(rellis):
    ds = Rellis3DDataset(rellis, keys=["lidar", "labels"])  # 2 sequences of 5
    cached = ds.cache()
    np.testing.assert_array_equal(cached.frame_sequence_ids, ds.frame_sequence_ids)
    np.testing.assert_array_equal(cached.frame_stems, ds.frame_stems)
    # frame_info through the cache keeps the right sequence, not one big blob
    assert cached.frame_info(6).sequence == ds.frame_info(6).sequence == "00001"


# -- H9: synchronize() must not collapse onto a single-event channel -----------


def test_synchronize_does_not_collapse_on_single_event_channel():
    from apairo.dataset.stream import StreamDataset

    ds = StreamDataset(
        {
            "oneshot": (np.array([0.5]), [np.zeros(1)]),
            "lidar": (np.arange(0.0, 1.0, 0.1), [np.zeros((3, 4)) for _ in range(10)]),
        }
    )
    view = ds.synchronize()  # reference=None -> lowest-frequency channel
    # Before the fix the one-shot (NaN frequency) was picked as the reference and
    # the view collapsed to its single frame; now 'lidar' is the reference.
    assert len(view) > 1


# -- H7: a per-frame .bin channel can be enumerated by a key/order regex --------


def test_bin_loader_accepts_frame_ordered_files(tmp_path):
    from apairo.loader.bin_loader import BINLoader

    for name in ["b.bin", "a.bin"]:
        np.arange(4, dtype=np.float32).tofile(tmp_path / name)
    ld = BINLoader(str(tmp_path), files=["a.bin", "b.bin"])  # frame order, not sorted
    assert ld.files == ["a.bin", "b.bin"]
    assert ld[0].shape == (1, 4)


# -- H13: loads_timestamps carries no hardcoded dataset-specific channel names --


def test_loads_timestamps_no_hardcoded_dataset_names(tmp_path):
    from apairo.loader import loads_timestamps

    d = tmp_path / "depth_left"  # a name the deleted TartanDrive map used to alias
    d.mkdir()
    np.save(d / "0.npy", np.zeros(3))  # data present, but no timestamps.txt
    with pytest.raises(ValueError, match="No timestamps.txt"):
        loads_timestamps(["depth_left"], {"depth_left": str(d)})


# -- H6/H8: timestamps_from a filename-key source, order-independently ----------


def test_timestamps_from_a_filename_key_source(tmp_path):
    from apairo.core.config import write_config
    from apairo.dataset.raw import RawDataset

    root = tmp_path / "seq"
    (root / "camera").mkdir(parents=True)
    (root / "lidar").mkdir(parents=True)
    for i in range(5):
        np.save(root / "camera" / f"frame{i:06d}-{i}.npy", np.zeros((2, 3), np.float32))
        np.save(root / "lidar" / f"{i:06d}.npy", np.zeros((2, 3), np.float32))
    write_config(
        root,
        {
            "version": 1,
            "channels": {
                "camera": {
                    "kind": "raw",
                    "loader": "npys",
                    "key": {"name": r"frame\d+-(\d+)"},
                },
                "lidar": {"kind": "raw", "loader": "npys", "timestamps_from": "camera"},
            },
        },
    )
    # 'lidar' is processed before 'camera' -- the order that used to crash because
    # camera's clock is filename-parsed (it has no timestamps.txt to read).
    ds = RawDataset(str(root), keys=["lidar", "camera"])
    np.testing.assert_array_equal(ds.timestamps["lidar"], np.arange(5, dtype=float))
    np.testing.assert_array_equal(ds.timestamps["camera"], np.arange(5, dtype=float))


# -- MEDIUM batch: M3, M4, M6, M7 ----------------------------------------------


def test_read_calibration_skips_malformed_transform(tmp_path):  # M3
    import yaml

    from apairo.core.config import read_calibration

    ap = tmp_path / ".apairo"
    ap.mkdir()
    (ap / "calibration.yaml").write_text(
        yaml.safe_dump({"transforms": {"base_to_lidar": {"parent": "b", "child": "l"}}})
    )
    cal = read_calibration(str(tmp_path))  # was KeyError('matrix')
    assert "base_to_lidar" not in cal


def test_verify_key_order_no_crash_on_unhashable_units(tmp_path):  # M4
    from apairo.core.config import verify_config, write_config

    write_config(
        tmp_path,
        {
            "version": 1,
            "channels": {
                "cam": {"loader": "img", "key": {"name": r"(\d+)", "units": [[0]]}}
            },
        },
    )
    issues = verify_config(str(tmp_path))  # was TypeError: unhashable type
    assert any("units" in s for s in issues)


def test_stream_frame_info_reports_channel_and_row():  # M6 (async)
    from apairo.dataset.stream import StreamDataset

    ds = StreamDataset(
        {
            "image": (np.array([0.0, 2.0]), [np.zeros(1), np.zeros(1)]),
            "lidar": (np.array([1.0, 3.0]), [np.zeros(3), np.zeros(3)]),
        }
    )
    fi = ds.frame_info(3)  # merged: image0, lidar0, image1, lidar1 -> 3 = lidar row 1
    assert fi.channel == "lidar" and fi.row == 1


def test_window_frame_info_uses_anchor_row(rellis):  # M6 (window)
    ds = Rellis3DDataset(rellis, keys=["lidar", "labels"])
    w = ds.window(size=2, stride=1, reduce=lambda samples: samples[-1])
    k = len(w) - 1
    # frame_info(k).row must be the anchor's parent row, not the window index
    assert w.frame_info(k).row == ds.frame_info(int(w.anchors[k])).row


def test_split_preserves_transforms(rellis):  # M7
    ds = Rellis3DDataset(rellis, keys=["lidar", "labels"]).transform(
        "lidar", lambda p: p[:1]
    )
    tr = ds.split("train")
    assert tr[0].data["lidar"].shape[0] == 1  # transform kept, not silently dropped


# -- Security (M5/L3) + robustness (L1/L2/M2) ----------------------------------


def test_safe_config_name_rejects_traversal():  # M5 / L3
    from apairo.core.config import safe_config_name

    assert safe_config_name("poses.npy") == "poses.npy"
    for bad in ["../secret", "/etc/passwd", "a/../../b"]:
        with pytest.raises(ValueError, match="relative path"):
            safe_config_name(bad)


def test_get_end_of_time_rejects_empty_clock():  # L1
    from apairo.utils.timestamps import get_end_of_time

    with pytest.raises(ValueError, match="empty clock"):
        get_end_of_time({"lidar": np.array([])})


def test_bin_loader_rejects_truncated_file(tmp_path):  # L2
    from apairo.loader.bin_loader import BINLoader

    (tmp_path / "0.bin").write_bytes(
        b"\x00" * (6 * 4)
    )  # 6 float32 -> not a multiple of 4
    with pytest.raises(ValueError, match="Corrupt/truncated"):
        BINLoader(str(tmp_path))[0]


def test_img_sort_key_numeric_first():  # M2
    from apairo.loader.img_loader import _img_sort_key

    names = ["10.jpg", "2.jpg", "1.jpg", "frame-x.jpg"]
    assert sorted(names, key=_img_sort_key) == [
        "1.jpg",
        "2.jpg",
        "10.jpg",
        "frame-x.jpg",
    ]


def test_img_loader_accepts_jpeg_bmp_uppercase(tmp_path):  # M2
    Image = pytest.importorskip("PIL.Image")
    for name in ["1.JPG", "10.jpeg", "2.bmp"]:
        Image.new("RGB", (2, 2)).save(tmp_path / name)
    from apairo.loader.img_loader import IMGLoader

    ld = IMGLoader(str(tmp_path))
    assert len(ld) == 3  # was 0 (case-sensitive png/jpg only) -> IndexError


# -- verify false-positives + directory-split filter (L4/L5/L6/M7b) -------------


def test_verify_calibration_accepts_null_distortion(tmp_path):  # L4
    import yaml

    from apairo.core.config import verify_calibration

    ap = tmp_path / ".apairo"
    ap.mkdir()
    (ap / "calibration.yaml").write_text(
        yaml.safe_dump(
            {"cameras": {"cam0": {"K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "D": None}}}
        )
    )
    assert not any("'D'" in s for s in verify_calibration(str(tmp_path)))


def test_verify_config_allows_self_alias(tmp_path):  # L5
    from apairo.core.config import verify_config, write_config

    write_config(
        tmp_path,
        {"version": 1, "channels": {"lidar": {"loader": "npys", "alias": "lidar"}}},
    )
    assert not any("collides" in s for s in verify_config(str(tmp_path)))


def test_verify_key_order_rejects_key_on_stacked_loader(tmp_path):  # L6
    from apairo.core.config import verify_config, write_config

    write_config(
        tmp_path,
        {
            "version": 1,
            "channels": {"poses": {"loader": "npy", "key": {"name": r"(\d+)"}}},
        },
    )
    assert any("per-frame loader" in s for s in verify_config(str(tmp_path)))


def test_filter_split_directory_split_preserves_transforms(tmp_path):  # M7b
    class _GooseDS(ProfiledDataset):
        _profile = "goose.yaml"

    for split, seq in [("train", "s0"), ("val", "s1")]:  # distinct seqs per split
        (tmp_path / "lidar" / split / seq).mkdir(parents=True)
        (tmp_path / "labels" / split / seq).mkdir(parents=True)
        for i in range(2):
            np.random.rand(4, 4).astype("f4").tofile(
                tmp_path / "lidar" / split / seq / f"{i:06d}.bin"
            )
            np.zeros(4, np.int32).tofile(
                tmp_path / "labels" / split / seq / f"{i:06d}.label"
            )
    ds = _GooseDS(str(tmp_path), keys=["lidar", "labels"]).transform(
        "lidar", lambda p: p[:1]
    )
    tr = ds.filter_split("train")  # was ValueError (no LST splits) before the fix
    assert len(tr) == 2  # only the train frames
    assert tr[0].data["lidar"].shape[0] == 1  # transform preserved mid-chain
