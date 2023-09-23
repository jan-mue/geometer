from collections.abc import Generator, Iterable
from typing import TypeVar

from geometer.utils.indexing import normalize_index, posify_index, sanitize_index
from geometer.utils.math import (
    adjugate,
    det,
    hat_matrix,
    inv,
    is_multiple,
    is_numerical_dtype,
    is_numerical_scalar,
    matmul,
    matvec,
    null_space,
    orth,
    outer,
    roots,
)

T = TypeVar("T")


def distinct(iterable: Iterable[T]) -> Generator[T, None, None]:
    """A simple generator that returns only the distinct elements of another iterable.

    Args:
        The iterable to filter.

    Yields:
        The distinct elements of the iterable.

    """
    seen = []
    for x in iterable:
        if x in seen:
            continue
        yield x
        seen.append(x)
