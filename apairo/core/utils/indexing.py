def from_end(idx: int, n: int) -> int:
    """Map a negative index onto ``[0, n)``, as a Python sequence does.

    ``-1`` is the last item. Below ``-n`` raises the ``IndexError`` a dataset
    raises past its end, with the range that negative indices allow.
    """
    if idx < -n:
        raise IndexError(f"Index {idx} out of range [{-n}, {n})")
    return idx + n
