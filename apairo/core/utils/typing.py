from typing import TypedDict, Union, SupportsFloat, Sequence
import numpy as np

_Key = str
_TimestampData = Union[np.ndarray, Sequence[SupportsFloat]]


class Timestamp(TypedDict):
    key: _Key
    timestamp: _TimestampData
