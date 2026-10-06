"""Standard datasets: the classes apairo selects by name, and the declarations
it ships.

A dataset with a fixed layout is a class: a YAML profile and a thin subclass
(SemanticKITTI, Rellis-3D, GOOSE, TartanDrive). A dataset read as it lies on
disk is a declaration, an ``apairo.yaml`` (TUM RGB-D, EuRoC MAV). Both ship with
apairo: adding one is a pull request, and ``pip install apairo[<dataset>]``
brings what reading it needs. A dataset kept in a package of its own registers
its class through the ``apairo.datasets`` entry point group and is then
selectable like a built-in one (``apairo init --as <Class>``):

.. code-block:: toml

    [project.entry-points."apairo.datasets"]
    ReassembleDataset = "apairo_reassemble:ReassembleDataset"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "apairo.datasets"
DECLARATIONS_DIR = Path(__file__).parent / "declarations"

_DATASETS: dict[str, type[Any]] = {}
_BUILTINS: frozenset[str] = frozenset()
_plugins_loaded = False


def register_dataset(cls: type[Any], *, replace: bool = False) -> type[Any]:
    """Register a dataset class under its class name. A name already taken is
    refused unless *replace* is set; a built-in dataset is never replaced."""
    name = cls.__name__
    if name in _DATASETS:
        if name in _BUILTINS:
            raise ValueError(f"'{name}' is a built-in dataset and cannot be replaced.")
        if not replace:
            raise ValueError(f"A dataset named '{name}' is already registered.")
    _DATASETS[name] = cls
    return cls


def _register_builtins() -> None:
    global _BUILTINS
    if _BUILTINS:
        return
    from apairo.dataset.goose import Goose3DDataset
    from apairo.dataset.raw import RawDataset
    from apairo.dataset.rellis import Rellis3DDataset
    from apairo.dataset.semantic_kitti import SemanticKittiDataset
    from apairo.dataset.tartan_kitti import TartanKittiDataset

    for cls in (
        RawDataset,
        SemanticKittiDataset,
        Rellis3DDataset,
        Goose3DDataset,
        TartanKittiDataset,
    ):
        _DATASETS.setdefault(cls.__name__, cls)
    _BUILTINS = frozenset(_DATASETS)


def _load_plugins() -> None:
    """Register the dataset classes installed through the ``apairo.datasets``
    entry point group, once. One that fails to load is skipped with a warning."""
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True
    from importlib.metadata import entry_points

    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            register_dataset(ep.load())
        except Exception as exc:
            logger.warning(
                "apairo: dataset plugin '%s' was not loaded: %s", ep.name, exc
            )


def _ready() -> None:
    _register_builtins()
    _load_plugins()


def get_dataset(name: str) -> type[Any]:
    """The registered dataset class called *name*."""
    _ready()
    try:
        return _DATASETS[name]
    except KeyError:
        raise KeyError(
            f"Unknown dataset '{name}'. Known: {', '.join(dataset_names())}."
        ) from None


def find_dataset(name: Any) -> type[Any] | None:
    """The registered dataset class called *name*, or ``None``."""
    _ready()
    return _DATASETS.get(name) if isinstance(name, str) else None


def dataset_names() -> list[str]:
    """Every registered dataset class name: ``RawDataset`` first, then the
    built-in ones, then the plugins'."""
    _ready()
    return list(_DATASETS)


# ── shipped declarations ─────────────────────────────────────────────────────


def declaration_names() -> list[str]:
    """The declarations that ship with apairo, by name (``tum_rgbd``, ...)."""
    return sorted(p.stem for p in DECLARATIONS_DIR.glob("*.yaml"))


def declaration(name: str) -> Path:
    """The path of the shipped declaration called *name*, for ``declare=``."""
    path = DECLARATIONS_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"No declaration named '{name}' ships with apairo. Shipped: "
            f"{', '.join(declaration_names())}."
        )
    return path


def resolve_declaration(value: str | Path | None) -> Path | None:
    """What a ``declare=`` value names: a declaration file, else the name of a
    declaration that ships with apairo. ``None`` stays ``None``."""
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_file():
        return path
    if (
        isinstance(value, str)
        and path.name == value
        and (DECLARATIONS_DIR / f"{value}.yaml").is_file()
    ):
        return declaration(value)
    raise FileNotFoundError(
        f"Declaration not found: '{value}' is neither a file nor the name of a "
        f"declaration that ships with apairo ({', '.join(declaration_names())})."
    )
