"""The chain behind a dataset, as a recipe: prune it, replay it, save it.

A dataset built in notebook style records how it was made: each method that
returns a new dataset stores the call on its result (:mod:`apairo.core._ops`)
and each dataset keeps its transform steps (``_pipeline``). Read back from
the final dataset, that is a recipe from one or more *sources* (datasets
built directly, such as a ``RawDataset`` or a ``StreamDataset``) to the
samples a model consumes.

:meth:`AbstractDataset.inference` turns that recipe into the chain to deploy:

- steps that do not lead to the requested outputs are pruned, from what each
  step declares it reads and writes (:func:`step_io`);
- steps that only belong in training declare ``inference = False`` on their
  class (augmentations) and are removed;
- frame filters, caches and repeats are curation, not computation: they are
  kept on the original data and dropped when the chain runs on another
  source (:meth:`Chain.on`).

The result is a :class:`InferenceView`, a dataset like any other, with a
:attr:`~InferenceView.summary` of what was kept, removed and flagged.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from apairo.core.abstract_dataset import (
    AbstractDataset,
    _ChannelStep,
    _PreprocessorStep,
)
from apairo.core.sample import Sample
from apairo.core.transform import Compose

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "Chain",
    "InferenceView",
    "Summary",
    "SummaryEntry",
    "step_io",
    "load_chain",
]

# Frame filters: curation of the original data, replayed there by index.
_FILTERS = frozenset({"filter", "filter_sequences", "filter_split"})
# Dropped wherever the chain runs: they change speed or epoch length, not values.
_PASSTHROUGH = frozenset({"cache", "repeat"})
_CAUSAL_STRATEGIES = frozenset({"previous", "latest"})

Channels = frozenset[str] | None  # None: any channel (unknown)


# --------------------------------------------------------------- step facts


def _name(fn: Any) -> str:
    """A readable label for a step or the callable inside it."""
    if isinstance(fn, _ChannelStep):
        return f"{_name(fn.fn)}({fn.key!r}" + (
            f" -> {fn.output!r})" if fn.output is not None else ")"
        )
    if isinstance(fn, _PreprocessorStep):
        return type(fn.preprocessor).__name__
    if isinstance(fn, Compose):
        return f"Compose([{', '.join(_name(t) for t in fn.transforms)}])"
    name = getattr(fn, "__name__", None)
    if name is None:
        return type(fn).__name__
    if name.startswith("<") or name in {"fn", "step", "inner", "wrapper"}:
        parts = getattr(fn, "__qualname__", "").split(".<locals>.")
        if len(parts) >= 2:
            return parts[-2].split(".")[-1]
    return name


def _declared(obj: Any, attr: str) -> frozenset[str] | None:
    keys = getattr(obj, attr, None)
    if isinstance(keys, str):
        return frozenset([keys])
    if isinstance(keys, (list, tuple, set, frozenset)):
        return frozenset(keys)
    return None


def _io(fn: Any) -> tuple[Channels, Channels]:
    """What a sample-level callable reads and writes, if it declares it."""
    if isinstance(fn, Compose):
        parts = [_io(t) for t in fn.transforms]
        if not parts or any(r is None or w is None for r, w in parts):
            return None, None
        reads: frozenset[str] = frozenset()
        writes: frozenset[str] = frozenset()
        for r, w in parts:
            reads |= r - writes  # type: ignore[operator]
            writes |= w  # type: ignore[operator]
        return reads, writes
    return _declared(fn, "input_keys"), _declared(fn, "output_keys")


def step_io(step: Callable) -> tuple[Channels, Channels]:
    """The channels a pipeline step reads and writes; ``None`` when unknown.

    - ``transform(key, fn, output=...)`` reads *key* and writes *output*
      (or *key*): known from the call itself.
    - ``transform(preprocessor)`` reads the preprocessor's ``input_keys`` and
      writes its outputs.
    - ``transform(fn)`` on a whole sample is opaque unless *fn* declares
      ``input_keys`` and ``output_keys`` (attributes or properties).
      A ``Compose`` of declared callables is declared.
    """
    if isinstance(step, _ChannelStep):
        return frozenset([step.key]), frozenset(
            [step.output if step.output is not None else step.key]
        )
    if isinstance(step, _PreprocessorStep):
        p = step.preprocessor
        outputs = [step.output] if step.output is not None else list(p.outputs)
        return frozenset(p.input_keys), frozenset(outputs)
    return _io(step)


def _training_only(fn: Any) -> bool:
    return getattr(fn, "inference", True) is False


def _strip_training(step: Callable) -> tuple[Callable | None, list[str]]:
    """*step* without its training-only parts, and the labels removed.

    ``None`` when the whole step is training-only. A ``Compose`` loses only
    the parts that are, so ``Compose([RangeFilter(), RandomRotation()])``
    keeps its filter.
    """

    def strip(fn: Any) -> tuple[Any | None, list[str]]:
        if _training_only(fn):
            return None, [_name(fn)]
        if isinstance(fn, Compose):
            kept, removed = [], []
            for part in fn.transforms:
                p, r = strip(part)
                removed += r
                if p is not None:
                    kept.append(p)
            if not removed:
                return fn, []
            return (Compose(kept) if kept else None), removed
        return fn, []

    if isinstance(step, _ChannelStep):
        if _training_only(step.fn):
            return None, [_name(step)]
        fn, removed = strip(step.fn)
        if fn is None:
            return None, removed
        return (step if fn is step.fn else _ChannelStep(step.key, fn, step.output)), (
            removed
        )
    if isinstance(step, _PreprocessorStep):
        return (
            (None, [_name(step)])
            if _training_only(step.preprocessor)
            else (
                step,
                [],
            )
        )
    return strip(step)


def _stateful(fn: Any) -> bool:
    inner = getattr(fn, "fn", None) or getattr(fn, "preprocessor", None) or fn
    return bool(getattr(inner, "stateful", False))


# ------------------------------------------------------------------ summary


@dataclass(frozen=True)
class SummaryEntry:
    """One line of a :class:`Summary`.

    Attributes:
        where: The dataset in the chain (``"RawDataset"``, ``"window"``, ...).
        item: The step or operation concerned.
        action: ``"kept"``, ``"removed"`` or ``"flag"``.
        reason: Why, in a few words.
    """

    where: str
    item: str
    action: str
    reason: str = ""


@dataclass
class Summary:
    """What :meth:`AbstractDataset.inference` kept, removed and flags.

    ``print(view.summary)`` lists every step; ``summary.removed`` and
    ``summary.flags`` give the entries to check before deploying.
    """

    outputs: list[str]
    entries: list[SummaryEntry] = field(default_factory=list)

    @property
    def kept(self) -> list[SummaryEntry]:
        return [e for e in self.entries if e.action == "kept"]

    @property
    def removed(self) -> list[SummaryEntry]:
        return [e for e in self.entries if e.action == "removed"]

    @property
    def flags(self) -> list[SummaryEntry]:
        return [e for e in self.entries if e.action == "flag"]

    def __str__(self) -> str:
        lines = [f"inference(outputs={self.outputs})"]
        marks = {"kept": "  kept    ", "removed": "  removed ", "flag": "  ⚠       "}
        for e in self.entries:
            reason = f"  -- {e.reason}" if e.reason else ""
            lines.append(f"{marks[e.action]}{e.where}: {e.item}{reason}")
        return "\n".join(lines)

    __repr__ = __str__


# ------------------------------------------------------------------- recipe


@dataclass(eq=False)
class _Node:
    """One dataset of the chain, detached from the live objects.

    A source has ``op is None``; any other node is ``inputs[0].<op>(...)``.
    ``steps`` and ``drop`` are its own transform pipeline.
    """

    op: str | None
    inputs: list[_Node]
    args: tuple
    kwargs: dict
    steps: list
    drop: set[str]
    label: str
    dataset: AbstractDataset | None = None  # the live source, on capture
    keys: list[str] | None = None  # a source's channels
    indices: np.ndarray | None = None  # a filter's kept frames
    opaque: bool = False  # a view built directly: its upstream is unknown
    required: Channels = None  # after pruning: what it must provide (None: any)


def _looks_like_view(ds: AbstractDataset) -> bool:
    return any(hasattr(ds, a) for a in ("_parent", "_datasets", "datasets", "_cache"))


def _resolved_reference(ds: AbstractDataset) -> str | None:
    ref = getattr(ds, "reference", None)
    if ref is None and getattr(ds, "datasets", None):
        ref = getattr(ds.datasets[0], "reference", None)  # type: ignore[attr-defined]
    return ref if isinstance(ref, str) else None


def _capture(ds: AbstractDataset, memo: dict[int, _Node]) -> _Node:
    """The recipe that produced *ds*, sharing nodes the chain shares."""
    if id(ds) in memo:
        return memo[id(ds)]
    if isinstance(ds, InferenceView):
        node = _capture(ds._inner, memo)
        node = _Node("select", [node], (list(ds.outputs),), {}, [], set(), "select")
        memo[id(ds)] = node
        return node
    op = getattr(ds, "_op", None)
    steps = list(getattr(ds, "_pipeline", []) or [])
    drop = set(getattr(ds, "_drop_keys", set()) or set())
    if op is None:
        node = _Node(
            None,
            [],
            (),
            {},
            steps,
            drop,
            type(ds).__name__,
            dataset=ds,
            keys=list(ds.keys),
            opaque=_looks_like_view(ds),
        )
    else:
        kwargs = dict(op.kwargs)
        if op.name == "synchronize":
            kwargs = _bind(op, "synchronize")
            if kwargs.get("reference") is None:
                kwargs["reference"] = _resolved_reference(ds)
                kwargs["_reference_resolved"] = True
        node = _Node(
            op.name,
            [_capture(i, memo) for i in op.inputs],
            op.args,
            kwargs,
            steps,
            drop,
            op.name,
            indices=getattr(ds, "indices", None) if op.name in _FILTERS else None,
        )
    memo[id(ds)] = node
    return node


def _bind(op: Any, name: str) -> dict:
    """*op*'s arguments by parameter name (``self`` and datasets excluded)."""
    method = getattr(type(op.inputs[0]), name)
    bound = inspect.signature(method).bind(op.inputs[0], *op.args, **op.kwargs)
    bound.apply_defaults()
    return {k: v for k, v in bound.arguments.items() if k != "self"}


def _topological(final: _Node) -> list[_Node]:
    """Nodes from *final* to the sources, every consumer before its inputs."""
    order: list[_Node] = []
    seen: set[int] = set()

    def visit(n: _Node) -> None:
        if id(n) in seen:
            return
        seen.add(id(n))
        for i in n.inputs:
            visit(i)
        order.append(n)

    visit(final)
    return order[::-1]


def _sources(final: _Node) -> list[_Node]:
    return [n for n in _topological(final) if n.op is None]


def _union(a: Channels, b: Channels) -> Channels:
    return None if a is None or b is None else a | b


def _param(node: _Node, name: str, position: int, default: Any = None) -> Any:
    if name in node.kwargs:
        return node.kwargs[name]
    return node.args[position] if len(node.args) > position else default


# ------------------------------------------------------------------ pruning


def _prune(final: _Node, outputs: list[str], report: list[SummaryEntry]) -> _Node:
    """A copy of the recipe keeping only what *outputs* need, outside training."""
    order = _topological(final)
    required: dict[int, Channels] = {id(final): frozenset(outputs)}
    pruned: dict[int, _Node] = {}
    per_node: dict[int, list[SummaryEntry]] = {}

    for node in order:
        need = required.get(id(node), frozenset())
        provided = need
        where = node.label
        kept: list = []
        local: list[SummaryEntry] = []
        for step in reversed(node.steps):
            step2, removed = _strip_training(step)
            for label in removed:
                local.append(SummaryEntry(where, label, "removed", "training only"))
            if step2 is None:
                continue
            reads, writes = step_io(step2)
            if writes is not None and need is not None and not (writes & need):
                local.append(
                    SummaryEntry(
                        where, _name(step2), "removed", "does not lead to the outputs"
                    )
                )
                continue
            kept.append(step2)
            local.append(SummaryEntry(where, _name(step2), "kept"))
            if _stateful(step2):
                local.append(
                    SummaryEntry(
                        where,
                        _name(step2),
                        "flag",
                        "keeps state: depends on read order, behaves differently "
                        "on a stream",
                    )
                )
            if reads is None or writes is None:
                local.append(
                    SummaryEntry(
                        where,
                        _name(step2),
                        "flag",
                        "opaque: declares no input_keys / output_keys, so every "
                        "step before it is kept",
                    )
                )
                need = None
            elif need is not None:
                need = (need - writes) | reads
        kept.reverse()
        local.reverse()
        per_node[id(node)] = local

        for i, inp in enumerate(node.inputs):
            need_in = _needed_by_op(node, need, i, local)
            required[id(inp)] = (
                need_in
                if id(inp) not in required
                else _union(required[id(inp)], need_in)
            )
        pruned[id(node)] = _Node(
            node.op,
            [],
            node.args,
            node.kwargs,
            kept,
            {k for k in node.drop if any(_writes(s, k) for s in kept)},
            node.label,
            dataset=node.dataset,
            keys=node.keys,
            indices=node.indices,
            opaque=node.opaque,
            required=need if node.op is None else provided,
        )

    for node in order:
        pruned[id(node)].inputs = [pruned[id(i)] for i in node.inputs]
    for node in reversed(order):  # sources first: the order the chain runs
        report.extend(per_node[id(node)])
    return pruned[id(final)]


def _writes(step: Callable, key: str) -> bool:
    w = step_io(step)[1]
    return w is None or key in w


def _needed_by_op(
    node: _Node, need: Channels, i: int, report: list[SummaryEntry]
) -> Channels:
    """What input *i* of *node* must provide for *node* to provide *need*."""
    if node.op == "select":
        keys = frozenset(_param(node, "keys", 0, []))
        return keys if need is None else need & keys
    if node.op == "window":
        reduce = _param(node, "reduce", 2)
        reads, _ = _io(reduce)
        if reads is None:
            report.append(
                SummaryEntry(
                    "window",
                    _name(reduce),
                    "flag",
                    "reducer declares no input_keys: every channel is kept upstream",
                )
            )
            return None
        return None if need is None else need | reads
    if node.op == "synchronize":
        # synchronize reads its parent's raw channels; the reference must stay.
        ref = node.kwargs.get("reference")
        if need is None or not isinstance(ref, str):
            return None
        return need | {ref}
    return need


# ---------------------------------------------------------------- flagging


def _flag_ops(final: _Node, report: list[SummaryEntry]) -> None:
    sources = _sources(final)
    for node in _topological(final):
        if node.op == "synchronize":
            ref = node.kwargs.get("reference")
            if node.kwargs.pop("_reference_resolved", False) and ref is not None:
                report.append(
                    SummaryEntry(
                        "synchronize",
                        f"reference={ref!r}",
                        "flag",
                        "resolved from reference=None (the lowest-rate channel); "
                        "fixed so another source uses the same clock",
                    )
                )
            if ref is not None and not isinstance(ref, str):
                report.append(
                    SummaryEntry(
                        "synchronize",
                        "reference=<array>",
                        "flag",
                        "an explicit clock cannot be replayed on another source",
                    )
                )
            for channel, why in _non_causal(node.kwargs.get("method", "previous")):
                report.append(
                    SummaryEntry(
                        "synchronize", f"{channel}: {why}", "flag", "non-causal"
                    )
                )
        elif node.op in _FILTERS:
            report.append(
                SummaryEntry(
                    node.op,
                    f"{len(node.indices) if node.indices is not None else '?'} frames",
                    "flag",
                    "kept on this data, dropped by .on() on another source",
                )
            )
        elif node.op in _PASSTHROUGH:
            report.append(
                SummaryEntry(node.op, node.op, "removed", "speed or epoch length only")
            )
        elif node.op == "join":
            for inp in node.inputs[1:]:
                if not set(map(id, _sources(inp))) & set(
                    map(id, _sources(node.inputs[0]))
                ):
                    report.append(
                        SummaryEntry(
                            "join",
                            " + ".join(s.label for s in _sources(inp)),
                            "flag",
                            "joins a second source: it must be available online",
                        )
                    )
    for s in sources:
        if s.opaque:
            report.append(
                SummaryEntry(
                    s.label,
                    s.label,
                    "flag",
                    "built directly, not through a dataset method: what is "
                    "upstream of it is not inspected",
                )
            )


def _non_causal(method: Any) -> list[tuple[str, str]]:
    from apairo.core.interpolator import Interpolator

    items = method.items() if isinstance(method, dict) else [("*", method)]
    out = []
    for channel, strategy in items:
        if isinstance(strategy, str):
            if strategy not in _CAUSAL_STRATEGIES:
                out.append((channel, f"{strategy!r} needs a later event"))
        elif isinstance(strategy, Interpolator):
            out.append(
                (
                    channel,
                    f"{type(strategy).__name__} interpolates up to the next event",
                )
            )
        else:
            out.append((channel, f"custom matching {_name(strategy)} (unknown)"))
    return out


# ------------------------------------------------------------------ replay


def _build(
    node: _Node,
    sources: dict[int, AbstractDataset],
    deploy: bool,
    memo: dict[int, AbstractDataset],
) -> AbstractDataset:
    """Run the recipe: each operation called again on rebuilt inputs."""
    if id(node) in memo:
        return memo[id(node)]
    if node.op is None:
        base = sources[id(node)]
        inherited = list(getattr(base, "_pipeline", []) or []) if deploy else []
        drop = set(getattr(base, "_drop_keys", set()) or set()) if deploy else set()
        out = base._branch()
        out._pipeline = inherited + list(node.steps)
        out._drop_keys = drop | node.drop
        memo[id(node)] = out
        return out

    ins = [_build(i, sources, deploy, memo) for i in node.inputs]
    if node.op in _FILTERS:
        if deploy:
            built = ins[0]
        else:
            from apairo.core.filtered_view import FilteredView

            built = FilteredView(ins[0], node.indices)
    elif node.op in _PASSTHROUGH:
        built = ins[0]
    elif node.op == "synchronize":
        kwargs = {k: v for k, v in node.kwargs.items() if not k.startswith("_")}
        built = ins[0].synchronize(**kwargs)
    else:
        method = getattr(ins[0], node.op)
        built = method(*ins[1:], *node.args, **node.kwargs)

    out = built._branch()
    out._pipeline = list(getattr(built, "_pipeline", []) or []) + list(node.steps)
    out._drop_keys = set(getattr(built, "_drop_keys", set()) or set()) | node.drop
    memo[id(node)] = out
    return out


def _provides(node: _Node) -> Channels:
    """The channels *node*'s samples hold, when it can be told."""
    if node.op is None:
        if node.opaque:
            return None
        have: Channels = frozenset(node.keys or [])
    elif node.op == "select":
        have = frozenset(_param(node, "keys", 0, []))
    elif node.op == "join":
        have = frozenset()
        for i in node.inputs:
            have = _union(have, _provides(i))
    elif node.op == "concat":
        parts = [_provides(i) for i in node.inputs]
        have = None if any(p is None for p in parts) else frozenset.intersection(*parts)  # type: ignore[arg-type]
    elif node.op == "window":
        _, writes = _io(_param(node, "reduce", 2))
        have = _union(_provides(node.inputs[0]), writes)
    else:
        have = _provides(node.inputs[0])
    for step in node.steps:
        have = _union(have, step_io(step)[1])
    return have


# -------------------------------------------------------------------- chain


class Chain:
    """A pruned recipe, ready to run on its original data or another source.

    Obtained from :attr:`InferenceView.chain` or :func:`load_chain`. Holds no
    data: :meth:`on` runs it on a source.
    """

    def __init__(self, final: _Node, outputs: list[str], summary: Summary) -> None:
        self._final = final
        self._outputs = list(outputs)
        self._summary = summary

    @property
    def outputs(self) -> list[str]:
        return list(self._outputs)

    @property
    def summary(self) -> Summary:
        return self._summary

    @property
    def sources(self) -> list[str]:
        """The class names of the sources the chain was built on."""
        return [s.label for s in _sources(self._final)]

    @property
    def reference(self) -> str | None:
        """The channel whose clock the chain follows: the reference of its
        ``synchronize`` closest to the source, or ``None`` without one (each
        event of the source is then a frame)."""
        for node in reversed(_topological(self._final)):  # sources first
            if node.op == "synchronize":
                ref = node.kwargs.get("reference")
                return ref if isinstance(ref, str) else None
        return None

    @property
    def lookback(self) -> int:
        """How many earlier reference frames one output can read: the reach of
        the chain's ``window`` operations (``(size - 1) * stride`` each). A
        stream executor keeps at least this many past frames."""
        reach = 0
        for node in _topological(self._final):
            if node.op == "window":
                size = int(_param(node, "size", 0, 1))
                stride = int(_param(node, "stride", 1, 1))
                reach += (size - 1) * stride
        return reach

    @property
    def causal(self) -> bool:
        """Whether every output depends on past events only."""
        return not any(e.reason == "non-causal" for e in self._summary.flags)

    def channels_needed(self) -> list[str] | None:
        """The channels a new source must provide; ``None`` when unknown."""
        (src,) = _sources(self._final)
        return sorted(src.required) if src.required is not None else src.keys

    def on(self, source: AbstractDataset) -> InferenceView:
        """Run the chain on *source* instead of the data it was built on.

        *source* must provide the channels the chain reads, under the same
        names (see :meth:`channels_needed`). Every operation is called again
        on it -- ``synchronize`` with the same reference and strategies,
        ``window`` with the same reducer -- and the steps registered on the
        original source are registered on *source*. Frame filters, caches and
        repeats are dropped: they selected or repeated frames of the original
        data.

        Raises:
            ValueError: If the chain has several sources (``join`` with a
                second dataset, ``concat``), was built on a view whose
                upstream is unknown, or *source* lacks a channel.
        """
        sources = _sources(self._final)
        if len(sources) != 1:
            raise ValueError(
                f"on() replaces the one source of a chain; this chain has "
                f"{len(sources)} ({', '.join(s.label for s in sources)}), through "
                f"join() or concat(). Run each branch's chain on its own source "
                f"and join the results."
            )
        (src,) = sources
        if src.opaque:
            raise ValueError(
                f"The chain starts at a {src.label} built directly, not through a "
                f"dataset method, so what produced it is unknown and cannot be "
                f"replayed. Build it with the dataset methods (filter, select, "
                f"synchronize, ...)."
            )
        for node in _topological(self._final):
            ref = node.kwargs.get("reference") if node.op == "synchronize" else None
            if ref is not None and not isinstance(ref, str):
                raise ValueError(
                    "synchronize() was given an explicit clock array: it belongs "
                    "to the original recording and cannot be replayed on another "
                    "source. Synchronize on a channel name."
                )
        needed = self.channels_needed() or []
        missing = sorted(set(needed) - set(source.keys))
        if missing:
            raise ValueError(
                f"The source lacks channels the chain reads: {missing} (it has "
                f"{sorted(source.keys)}). Name the source's channels as the "
                f"training data's, e.g. a topic mapping."
            )
        inner = _build(self._final, {id(src): source}, deploy=True, memo={})
        return InferenceView(inner, self._outputs, self)

    def save(self, path: str | Path) -> Path:
        """Write the chain as a JSON declaration (see :mod:`apairo.core.chain_io`)."""
        from apairo.core.chain_io import save_chain

        return save_chain(self, path)

    def __repr__(self) -> str:
        return f"Chain(outputs={self._outputs}, sources={self.sources})"


def load_chain(path: str | Path, *, trusted: Iterable[str] = ()) -> Chain:
    """Read a chain written by :meth:`Chain.save` -- see :mod:`apairo.core.chain_io`."""
    from apairo.core.chain_io import load_chain as _load

    return _load(path, trusted=trusted)


def inference(
    ds: AbstractDataset, outputs: str | list[str], causal: bool = False
) -> InferenceView:
    """Implementation of :meth:`AbstractDataset.inference`."""
    outputs = [outputs] if isinstance(outputs, str) else list(outputs)
    if not outputs:
        raise ValueError("inference() needs at least one output channel.")
    captured = _capture(ds, {})
    have = _provides(captured)
    if have is not None:
        unknown = [o for o in outputs if o not in have]
        if unknown:
            raise ValueError(
                f"inference(outputs=...) names channels the chain does not "
                f"produce: {unknown}. Available: {sorted(have)}."
            )
    entries: list[SummaryEntry] = []
    final = _prune(captured, outputs, entries)
    _flag_ops(final, entries)
    summary = Summary(outputs, entries)
    if causal:
        non_causal = [e for e in summary.flags if e.reason == "non-causal"]
        if non_causal:
            raise ValueError(
                "inference(causal=True): the chain needs future events: "
                + "; ".join(e.item for e in non_causal)
                + ". Use method='previous' for these channels."
            )
    sources = {id(n): n.dataset for n in _sources(final)}
    inner = _build(final, sources, deploy=False, memo={})  # type: ignore[arg-type]
    return InferenceView(inner, outputs, Chain(final, outputs, summary))


# ------------------------------------------------------------- the dataset


class InferenceView(AbstractDataset):
    """The chain of a dataset as deployed: only what *outputs* need, without
    training-only steps. Returned by :meth:`AbstractDataset.inference`.

    A full dataset: index it, iterate it, hand it to a ``DataLoader``. Its
    samples hold the requested outputs only. :meth:`on` runs the same chain
    on another source; :meth:`save` writes it beside a model checkpoint.
    """

    def __init__(
        self, inner: AbstractDataset, outputs: list[str], chain: Chain
    ) -> None:
        self._inner = inner
        self._outputs = list(outputs)
        self._chain = chain

    @property
    def outputs(self) -> list[str]:
        return list(self._outputs)

    @property
    def keys(self) -> list[str]:
        return list(self._outputs)

    @keys.setter
    def keys(self, keys) -> None:
        raise AttributeError("An inference view's channels are its outputs.")

    @property
    def summary(self) -> Summary:
        """What the chain kept, removed and flags -- ``print()`` it."""
        return self._chain.summary

    @property
    def chain(self) -> Chain:
        return self._chain

    @property
    def is_synchronous(self) -> bool:
        return self._inner.is_synchronous

    @property
    def root_dir(self):
        return getattr(self._inner, "root_dir", None)

    @property
    def frame_sequence_ids(self) -> np.ndarray:
        return self._inner.frame_sequence_ids

    @property
    def frame_stems(self) -> np.ndarray:
        return self._inner.frame_stems

    @property
    def frame_channel_ids(self) -> np.ndarray:
        return self._inner.frame_channel_ids

    def frame_info(self, idx: int):
        return self._inner.frame_info(idx)

    def on(self, source: AbstractDataset) -> InferenceView:
        """The same chain on another source -- see :meth:`Chain.on`."""
        return self._chain.on(source)

    def save(self, path: str | Path) -> Path:
        """Write the chain as a declaration -- see :meth:`Chain.save`."""
        return self._chain.save(path)

    def __len__(self) -> int:
        return len(self._inner)

    def _load(self, idx: int) -> Sample:
        sample = self._inner[idx]
        missing = [k for k in self._outputs if k not in sample.data]
        if missing:
            raise KeyError(
                f"inference outputs {missing} are absent from the sample (it "
                f"holds {sorted(sample.data)})."
            )
        return Sample(
            data={k: sample.data[k] for k in self._outputs},
            timestamp=sample.timestamp,
        )

    def __repr__(self) -> str:
        return (
            f"InferenceView(n={len(self)}, outputs={self._outputs}, "
            f"sources={self._chain.sources})"
        )
