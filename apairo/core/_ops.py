"""Provenance of a dataset: the method call that produced it.

Every dataset method that returns a new dataset (``synchronize``, ``window``,
``select``, ``filter``, ``cache``, ``join``, ...) records on its result the
call that built it: the method name, the input datasets and the arguments.
Following ``_op`` from a dataset back to datasets that have none gives the
chain as a recipe, which :mod:`apairo.core.chain` prunes, replays on another
source and saves. Nothing is loaded and nothing is copied: ``_op`` only holds
references.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, NamedTuple, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class Op(NamedTuple):
    """The call that produced a dataset: ``inputs[0].<name>(*inputs[1:]...)``.

    Attributes:
        name: Method name (``"synchronize"``, ``"window"``, ...).
        inputs: The dataset the method was called on, then any dataset passed
            as an argument (``join`` / ``concat`` operands).
        args: Remaining positional arguments, datasets excluded.
        kwargs: Keyword arguments.
    """

    name: str
    inputs: tuple
    args: tuple
    kwargs: dict


def records(name: str) -> Callable[[F], F]:
    """Record the call on the dataset a method returns, as ``result._op``.

    Applied to every method that returns a *new* dataset. The outermost
    recorded call wins: ``root.synchronize`` builds per-sequence views through
    inner ``synchronize`` calls, then records itself on the concatenation it
    returns, which is the call the user made.
    """

    def decorate(method: F) -> F:
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            out = method(self, *args, **kwargs)
            from apairo.core.abstract_dataset import AbstractDataset

            if isinstance(out, AbstractDataset) and out is not self:
                datasets = tuple(a for a in args if isinstance(a, AbstractDataset))
                rest = tuple(a for a in args if not isinstance(a, AbstractDataset))
                out._op = Op(name, (self, *datasets), rest, dict(kwargs))
            return out

        return wrapper  # type: ignore[return-value]

    return decorate
