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
