"""Reusable file-naming policies.

A naming policy maps a file or archive-member *name* to a frame index
(or builds a name from an index).  Loaders and writers never parse names
themselves — the dataset imposes a policy, typically one of these.
"""

from __future__ import annotations
import os
import re
from typing import Optional

_INTEGER_NAME = re.compile(r"^0*([0-9]+)\.[A-Za-z0-9]+$")


def integer_frame_index(name: str) -> Optional[int]:
    """``<int>.<ext>`` basename (zero-padding accepted) -> frame index.

    ``000042.jpg`` -> 42, ``7.png`` -> 7, ``frame_7.png`` -> ``None``.
    """
    match = _INTEGER_NAME.match(os.path.basename(name))
    return int(match.group(1)) if match else None
