"""Save a chain as a JSON declaration, and load it back.

A deployed chain travels beside its model checkpoint. It is written as a
*declaration*, not a pickle: for each step, the import path of its class and
the values that rebuild it. Loading imports those classes and sets those
values; it never runs code stored in the file, and it only imports from
trusted packages (apairo and its satellites, plus the ones you name).

What can be declared:

- numbers, strings, booleans, ``None``, lists, tuples, dicts with string
  keys, numpy arrays and scalars;
- functions defined at module level (not lambdas, not closures), and
  ``functools.partial`` of them;
- instances of classes defined at module level. An object with a
  ``get_config()`` method is rebuilt as ``cls(**get_config())``; any other
  is rebuilt from its attributes, which must themselves be declarable.

A step holding something else -- a model's weights, a random generator, a
lambda -- makes :func:`save_chain` refuse and name it. Give its class a
``get_config()`` returning what rebuilds it (a checkpoint path, say).
"""

from __future__ import annotations

import functools
import importlib
import json
import types
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from apairo.core.abstract_dataset import _ChannelStep, _PreprocessorStep
from apairo.core.transform import Compose

if TYPE_CHECKING:
    from apairo.core.chain import Chain, _Node

FORMAT = "apairo-chain"
VERSION = 1
TRUSTED = ("apairo", "apairo_transform", "apairo_preprocess", "apairo_ros")
_TAG = "__apairo__"
_HEAPTYPE = 1 << 9  # Py_TPFLAGS_HEAPTYPE: a class defined in Python


class ChainDeclarationError(ValueError):
    """A chain holds something that cannot be written as a declaration, or a
    file names something that may not be loaded."""


# ----------------------------------------------------------------- writing


def _path_of(obj: Any, where: str) -> str:
    cls = obj if isinstance(obj, (type, types.FunctionType)) else type(obj)
    module, qualname = cls.__module__, cls.__qualname__
    if "<locals>" in qualname or qualname == "<lambda>":
        raise ChainDeclarationError(
            f"{where}: {qualname} is defined inside a function (a lambda or a "
            f"closure), so it cannot be imported back. Define it at module level."
        )
    if module == "__main__":
        raise ChainDeclarationError(
            f"{where}: {qualname} is defined in the script being run (__main__), "
            f"which another process cannot import. Move it to a module."
        )
    return f"{module}:{qualname}"


def _encode(value: Any, where: str) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            raise ChainDeclarationError(f"{where}: an object array cannot be declared.")
        return {_TAG: "ndarray", "dtype": str(value.dtype), "data": value.tolist()}
    if isinstance(value, list):
        return [_encode(v, f"{where}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, tuple):
        return {
            _TAG: "tuple",
            "items": [_encode(v, f"{where}[{i}]") for i, v in enumerate(value)],
        }
    if isinstance(value, (set, frozenset)):
        return {_TAG: "set", "items": sorted(value)}
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise ChainDeclarationError(f"{where}: dict keys must be strings.")
        return {
            _TAG: "dict",
            "items": {k: _encode(v, f"{where}[{k!r}]") for k, v in value.items()},
        }
    if isinstance(value, _ChannelStep):
        return {
            _TAG: "channel_step",
            "key": value.key,
            "output": value.output,
            "fn": _encode(value.fn, f"{where}.fn"),
        }
    if isinstance(value, _PreprocessorStep):
        return {
            _TAG: "preprocessor_step",
            "output": value.output,
            "preprocessor": _encode(value.preprocessor, f"{where}.preprocessor"),
        }
    if isinstance(value, Compose):
        return {
            _TAG: "compose",
            "transforms": [
                _encode(t, f"{where}.transforms[{i}]")
                for i, t in enumerate(value.transforms)
            ],
        }
    if isinstance(value, types.FunctionType):
        return {_TAG: "function", "path": _path_of(value, where)}
    if isinstance(value, functools.partial):
        return {
            _TAG: "partial",
            "func": _encode(value.func, f"{where}.func"),
            "args": [_encode(a, f"{where}.args") for a in value.args],
            "keywords": _encode(dict(value.keywords), f"{where}.keywords"),
        }
    return _encode_object(value, where)


def _encode_object(obj: Any, where: str) -> dict:
    name = type(obj).__name__
    get_config = getattr(obj, "get_config", None)
    if callable(get_config):
        return {
            _TAG: "object",
            "path": _path_of(obj, where),
            "config": _encode(get_config(), f"{where} ({name}.get_config())"),
        }
    if not type(obj).__flags__ & _HEAPTYPE:
        # Implemented in C: its state is not in its attributes.
        raise ChainDeclarationError(
            f"{where}: a {type(obj).__module__}.{name} cannot be declared. Give the "
            f"class that holds it a get_config() returning what rebuilds it."
        )
    state: dict[str, Any] | None = None
    if hasattr(obj, "__dict__"):
        state = dict(vars(obj))
    slots = [
        s for cls in type(obj).__mro__ for s in getattr(cls, "__slots__", ()) or ()
    ]
    if slots:
        state = state or {}
        state.update({s: getattr(obj, s) for s in slots if hasattr(obj, s)})
    if state is None:
        raise ChainDeclarationError(
            f"{where}: a {type(obj).__module__}.{name} cannot be declared. Give the "
            f"class that holds it a get_config() returning what rebuilds it."
        )
    path = _path_of(obj, where)
    return {
        _TAG: "object",
        "path": path,
        "state": {k: _encode(v, f"{where} ({name}.{k})") for k, v in state.items()},
    }


def _versions(paths: Iterable[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in paths:
        top = path.split(":")[0].split(".")[0]
        if top in out:
            continue
        try:
            out[top] = str(getattr(importlib.import_module(top), "__version__", ""))
        except ImportError:
            out[top] = ""
    return out


def _collect_paths(value: Any, acc: set[str]) -> None:
    if isinstance(value, dict):
        if "path" in value and value.get(_TAG) in ("object", "function"):
            acc.add(value["path"])
        for v in value.values():
            _collect_paths(v, acc)
    elif isinstance(value, list):
        for v in value:
            _collect_paths(v, acc)


def save_chain(chain: Chain, path: str | Path) -> Path:
    """Write *chain* to *path* as JSON. See the module docstring.

    Raises:
        ChainDeclarationError: If a step or argument cannot be declared; the
            message names it and the fix.
    """
    from apairo.core.chain import _topological

    nodes = _topological(chain._final)[::-1]  # sources first
    ids = {id(n): i for i, n in enumerate(nodes)}
    out_nodes = []
    for i, n in enumerate(nodes):
        where = f"{n.label} (node {i})"
        kwargs = {k: v for k, v in n.kwargs.items() if not k.startswith("_")}
        out_nodes.append(
            {
                "op": n.op,
                "label": n.label,
                "inputs": [ids[id(x)] for x in n.inputs],
                "args": _encode(tuple(n.args), f"{where} args"),
                "kwargs": _encode(kwargs, f"{where} kwargs"),
                "steps": [
                    _encode(s, f"{where} step {j}") for j, s in enumerate(n.steps)
                ],
                "drop": sorted(n.drop),
                "keys": n.keys,
                "required": sorted(n.required) if n.required is not None else None,
            }
        )
    paths: set[str] = set()
    _collect_paths(out_nodes, paths)
    doc = {
        "format": FORMAT,
        "version": VERSION,
        "packages": _versions(["apairo:", *sorted(paths)]),
        "outputs": chain.outputs,
        "final": ids[id(chain._final)],
        "nodes": out_nodes,
        "summary": [
            {"where": e.where, "item": e.item, "action": e.action, "reason": e.reason}
            for e in chain.summary.entries
        ],
    }
    path = Path(path)
    path.write_text(json.dumps(doc, indent=1, allow_nan=True), encoding="utf-8")
    return path


# ----------------------------------------------------------------- reading


def _import(path: str, trusted: tuple[str, ...]) -> Any:
    module, _, qualname = path.partition(":")
    if not any(module == t or module.startswith(t + ".") for t in trusted):
        raise ChainDeclarationError(
            f"The chain names {path}, outside the trusted packages {list(trusted)}. "
            f"If you trust it, pass load_chain(..., trusted=[{module.split('.')[0]!r}])."
        )
    obj: Any = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _decode(value: Any, trusted: tuple[str, ...]) -> Any:
    if isinstance(value, list):
        return [_decode(v, trusted) for v in value]
    if not isinstance(value, dict):
        return value
    tag = value.get(_TAG)
    if tag == "ndarray":
        return np.asarray(value["data"], dtype=value["dtype"])
    if tag == "tuple":
        return tuple(_decode(v, trusted) for v in value["items"])
    if tag == "set":
        return set(value["items"])
    if tag == "dict":
        return {k: _decode(v, trusted) for k, v in value["items"].items()}
    if tag == "channel_step":
        return _ChannelStep(
            value["key"], _decode(value["fn"], trusted), value["output"]
        )
    if tag == "preprocessor_step":
        return _PreprocessorStep(
            _decode(value["preprocessor"], trusted), value["output"]
        )
    if tag == "compose":
        return Compose([_decode(t, trusted) for t in value["transforms"]])
    if tag == "function":
        return _import(value["path"], trusted)
    if tag == "partial":
        return functools.partial(
            _decode(value["func"], trusted),
            *[_decode(a, trusted) for a in value["args"]],
            **_decode(value["keywords"], trusted),
        )
    if tag == "object":
        cls = _import(value["path"], trusted)
        if "config" in value:
            return cls(**_decode(value["config"], trusted))
        obj = cls.__new__(cls)
        for k, v in value["state"].items():
            object.__setattr__(obj, k, _decode(v, trusted))
        return obj
    raise ChainDeclarationError(f"Unknown entry in the chain file: {value!r:.80}")


def load_chain(path: str | Path, *, trusted: Iterable[str] = ()) -> Chain:
    """Read a chain written by :func:`save_chain`; run it with ``.on(source)``.

    Args:
        path: The JSON file.
        trusted: Packages, besides apairo and its satellites, whose classes
            the file may name (the package holding your model step, say).

    Raises:
        ChainDeclarationError: If the file is not a chain, or names a class
            outside the trusted packages.
    """
    from apairo.core.chain import Chain, Summary, SummaryEntry, _Node

    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("format") != FORMAT:
        raise ChainDeclarationError(f"{path} is not an apairo chain file.")
    if doc.get("version", 0) > VERSION:
        raise ChainDeclarationError(
            f"{path} was written by a newer apairo (chain format "
            f"{doc['version']}, this one reads {VERSION}). Upgrade apairo."
        )
    allowed = TRUSTED + tuple(trusted)
    nodes: list[_Node] = []
    for raw in doc["nodes"]:
        nodes.append(
            _Node(
                raw["op"],
                [nodes[i] for i in raw["inputs"]],
                _decode(raw["args"], allowed),
                _decode(raw["kwargs"], allowed),
                [_decode(s, allowed) for s in raw["steps"]],
                set(raw["drop"]),
                raw["label"],
                keys=raw["keys"],
                required=(
                    frozenset(raw["required"]) if raw["required"] is not None else None
                ),
            )
        )
    _warn_versions(doc.get("packages", {}), path)
    summary = Summary(
        list(doc["outputs"]), [SummaryEntry(**e) for e in doc.get("summary", [])]
    )
    return Chain(nodes[doc["final"]], doc["outputs"], summary)


def _warn_versions(recorded: dict[str, str], path: str | Path) -> None:
    differ = []
    for top, version in recorded.items():
        try:
            now = str(getattr(importlib.import_module(top), "__version__", ""))
        except ImportError:
            now = "not installed"
        if version and now != version:
            differ.append(f"{top} {version} -> {now}")
    if differ:
        warnings.warn(
            f"{path} was saved with other package versions ({', '.join(differ)}); "
            f"check the chain still gives the same outputs (replay a recording).",
            UserWarning,
            stacklevel=3,
        )
