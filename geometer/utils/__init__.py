from .indexing import normalize_index, posify_index, sanitize_index  # noqa: F401
from .math import adjugate, det, hat_matrix, inv, is_multiple, matmul, matvec, null_space, orth, roots  # noqa: F401


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
