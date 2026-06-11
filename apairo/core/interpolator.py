from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Interpolator(ABC):
    """Synthesize a channel value at the reference instant from its two
    bracketing events.

    This is the *value-level* counterpart of the index-level matching
    strategies (``"latest"``, ``"nearest"``): instead of picking an existing
    event, an interpolator builds a new value at ``t_ref``.  Use it for
    continuous signals -- poses, IMU, commands -- never for data that cannot
    be blended (point clouds, images).

    Contract, as orchestrated by
    :class:`~apairo.core.synchronized_view.SynchronizedView`:

    * a channel whose strategy is an ``Interpolator`` receives, for each
      reference tick ``t``, its two bracketing events ``(t0, v0)`` and
      ``(t1, v1)`` with ``t0 <= t <= t1`` and ``t0 < t1``;
    * ticks not bracketed by two events (before the first event or after the
      last) are dropped from the view;
    * exact matches (``t == t0``) bypass the interpolator -- the stored
      value is returned directly, so implementations never see ``t0 == t1``;
    * with ``tolerance``, *both* neighbours must lie within tolerance of
      ``t`` (``max(t - t0, t1 - t) <= tolerance``).

    Concrete implementations (``LinearInterp``, ``Se3Interp``, ...) live in
    ``apairo_transform.interp``.

    Example::

        class LinearInterp(Interpolator):
            def __call__(self, t, t0, v0, t1, v1):
                a = (t - t0) / (t1 - t0)
                return (1.0 - a) * v0 + a * v1
    """

    @abstractmethod
    def __call__(self, t: float, t0: float, v0: Any, t1: float, v1: Any) -> Any:
        """Return the channel value at time *t*, ``t0 <= t <= t1``, ``t0 < t1``."""
        ...
