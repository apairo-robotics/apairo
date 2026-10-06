"""The standard datasets apairo knows by name, and the ones a package adds.

The datasets that ship with apairo -- profiled classes and declarations --
are selected by name. A dataset kept in its own package registers its class
through the ``apairo.datasets`` entry point group and becomes selectable the
same way (``apairo init --as <Class>``), with no line of apairo changed.
"""

from __future__ import annotations

import logging
from importlib.metadata import EntryPoint

import numpy as np
import pytest

import apairo.dataset.registry as registry
from apairo.cli import main
from apairo.dataset.raw import RawDataset
from apairo.dataset.registry import (
    declaration,
    declaration_names,
    get_dataset,
    register_dataset,
)


class LabDataset(RawDataset):
    """A lab's recordings, as a package of its own would ship them."""


@pytest.fixture
def installed(monkeypatch):
    registry.dataset_names()  # built-ins in first
    monkeypatch.setattr(registry, "_DATASETS", dict(registry._DATASETS))

    def install(*specs: tuple[str, str]) -> None:
        eps = [EntryPoint(n, v, registry.ENTRY_POINT_GROUP) for n, v in specs]
        monkeypatch.setattr(
            "importlib.metadata.entry_points",
            lambda group=None: [ep for ep in eps if ep.group == group],
        )
        monkeypatch.setattr(registry, "_plugins_loaded", False)

    return install


def test_the_standard_datasets_are_registered():
    names = registry.dataset_names()
    assert names[0] == "RawDataset"
    assert {
        "SemanticKittiDataset",
        "Rellis3DDataset",
        "Goose3DDataset",
        "TartanKittiDataset",
    } <= set(names)


def test_the_standard_declarations_ship_by_name():
    assert {"tum_rgbd", "euroc_vicon_room", "euroc_machine_hall"} <= set(
        declaration_names()
    )
    assert declaration("tum_rgbd").is_file()
    with pytest.raises(FileNotFoundError, match="Shipped: euroc_machine_hall"):
        declaration("kitti360")


def test_a_dataset_package_is_selectable_by_name(tmp_path, installed, capsys):
    installed(("LabDataset", "test.dataset.test_registry:LabDataset"))
    assert get_dataset("LabDataset") is LabDataset

    seq = tmp_path / "seq" / "lidar"
    seq.mkdir(parents=True)
    for i in range(3):
        np.save(seq / f"{i:06d}.npy", np.zeros(2))
    np.savetxt(seq / "timestamps.txt", [0.0, 0.1, 0.2])
    try:
        code = main(["init", str(seq.parent), "--as", "LabDataset"])
    except SystemExit as exc:
        code = exc.code
    assert code in (0, None), capsys.readouterr().out
    assert LabDataset(seq.parent).keys == ["lidar"]


def test_a_standard_dataset_cannot_be_replaced(installed):
    installed()

    class Goose3DDataset(RawDataset):
        pass

    with pytest.raises(ValueError, match="built-in dataset and cannot be replaced"):
        register_dataset(Goose3DDataset)


def test_a_broken_dataset_package_is_skipped_with_a_warning(installed, caplog):
    installed(
        ("Missing", "apairo_missing_module:Missing"),
        ("LabDataset", "test.dataset.test_registry:LabDataset"),
    )
    with caplog.at_level(logging.WARNING, logger="apairo.dataset.registry"):
        names = registry.dataset_names()
    assert "LabDataset" in names and "Missing" not in names
    assert "dataset plugin 'Missing' was not loaded" in caplog.text
