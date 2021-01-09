from .math import (
    null_space,
    hat_matrix,
    is_multiple,
    orth,
    adjugate,
    det,
    inv,
    matmul,
    matvec,
    roots,
)
from .indexing import posify_index, normalize_index, sanitize_index


def distinct(iterable):
    """A simple generator that returns only the distinct elements of another iterable.

    Parameters
    ----------
    iterable
        The iterable to filter.

    Yields
    ------
        The distinct elements of the iterable.

    """
    seen = []
    for x in iterable:
        if x in seen:
            continue
        yield x
        seen.append(x)
