from typing import Any, Callable
from torchvision.transforms import Normalize


class IllegalArgumentError(ValueError):
    ...


class Node:
    r"""A node of the pipeline.

    """

    def __init__(
        self,
        func: Callable,
        inputs: str | list[str] | dict[str, str] = None,
        outputs: str | list[str] | dict[str, str] = None,
    ):
        """Create a transform.
        """
        if not callable(func):
            raise IllegalArgumentError
        self._func = func

        self._inputs = inputs
        self._outputs = outputs

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        r"""Apply the transform to args given.
        """
        return self._func(*args)


class IdentityNode(Node):
    r"""Identity :class:`Transform` class.

    Only for debugging
    """

    def __call__(self, x):
        return x


class NormalizeNode(Node):
    """"""

    def __init__(self, mean, std, inputs, outputs):
        func = Normalize(mean, std)
        func.__call__ = func.forward
        super().__init__(func, inputs, outputs)

    @property
    def func(self):
        return self._func

    @property
    def mean(self):
        return self.func.mean

    @property
    def std(self):
        return self.func.std

    @mean.setter
    def mean(self, mean):
        self._func.mean = mean

    @std.setter
    def std(self, std):
        self._func.std = std
