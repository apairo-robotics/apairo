from collections.abc import Sequence
from typing import SupportsFloat, TypedDict

import numpy as np

_Key = str
_TimestampData = np.ndarray | Sequence[SupportsFloat]


class Timestamp(TypedDict):
    key: _Key
    timestamp: _TimestampData
