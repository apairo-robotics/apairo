"""The sidecar files are written atomically, and registrations made in
parallel do not lose each other's entry."""

from __future__ import annotations

import threading

import pytest
import yaml

from apairo.core import config
from apairo.core.config import (
    config_lock,
    read_config,
    register_channel,
    write_config,
)


def test_a_failed_write_keeps_the_previous_file(tmp_path, monkeypatch):
    write_config(tmp_path, {"version": 1, "channels": {"a": {"loader": "npy"}}})

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(yaml, "dump", boom)
    with pytest.raises(RuntimeError):
        write_config(tmp_path, {"version": 1, "channels": {}})
    monkeypatch.undo()
    assert read_config(tmp_path)["channels"] == {"a": {"loader": "npy"}}
    assert [p.name for p in (tmp_path / ".apairo").iterdir()] == ["channels.yaml"]


def test_parallel_registrations_all_land(tmp_path):
    write_config(tmp_path, {"version": 1, "channels": {}})
    threads = [
        threading.Thread(target=register_channel, args=(tmp_path, f"k{i}", "npy"))
        for i in range(24)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(read_config(tmp_path)["channels"]) == sorted(
        f"k{i}" for i in range(24)
    )
    assert not (tmp_path / ".apairo" / ".lock").exists()


def test_a_held_lock_times_out_and_a_stale_one_is_taken_over(tmp_path, monkeypatch):
    with config_lock(tmp_path):
        with pytest.raises(TimeoutError, match="held by another process"):
            with config_lock(tmp_path, timeout=0.1):
                pass
    (tmp_path / ".apairo" / ".lock").touch()  # left by a killed process
    monkeypatch.setattr(config, "_LOCK_STALE_S", 0.0)
    with config_lock(tmp_path, timeout=1.0):
        pass


# ----------------------------------------------- removing a channel's data


def _dataset(tmp_path, channels):
    root = tmp_path / "ds"
    root.mkdir()
    write_config(root, {"version": 1, "channels": channels})
    return root


@pytest.mark.parametrize("name", ["../victim", "/abs/victim", "a/../../victim"])
def test_removing_data_never_leaves_the_dataset(tmp_path, name):
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.npy").write_bytes(b"x")
    root = _dataset(tmp_path, {name: {"kind": "raw", "loader": "npys"}})
    with pytest.raises(ValueError, match="relative path|outside the dataset"):
        config.remove_channel(root, name, data=True)
    assert (victim / "keep.npy").exists()
    assert name in read_config(root)["channels"]  # the refusal changed nothing


def test_a_symlinked_channel_directory_is_not_followed_out(tmp_path):
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.npy").write_bytes(b"x")
    root = _dataset(tmp_path, {"lidar": {"kind": "raw", "loader": "npys"}})
    (root / "lidar").symlink_to(victim, target_is_directory=True)
    with pytest.raises(ValueError, match="outside the dataset"):
        config.remove_channel(root, "lidar", data=True)
    assert (victim / "keep.npy").exists()


def test_a_suffixed_channel_loses_only_its_own_files(tmp_path):
    root = _dataset(
        tmp_path,
        {
            "lidar": {"kind": "raw", "loader": "npys"},
            "lidar_intensity": {
                "kind": "raw",
                "loader": "npys",
                "directory": "lidar",
                "suffix": "intensity",
            },
            "evil": {"kind": "raw", "loader": "npys", "directory": "..", "suffix": "x"},
        },
    )
    (root / "lidar").mkdir()
    for name in ("000000.npy", "000000_intensity.npy", "000000_intensity.png"):
        (root / "lidar" / name).write_bytes(b"x")
    config.remove_channel(root, "lidar_intensity", data=True)
    assert sorted(p.name for p in (root / "lidar").iterdir()) == ["000000.npy"]
    with pytest.raises(ValueError, match="relative path"):
        config.remove_channel(root, "evil", data=True)
