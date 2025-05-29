from __future__ import annotations

from collections.abc import Iterable, Sequence
from numbers import Number
from typing import TYPE_CHECKING, Literal, TypedDict, Union

import numpy as np
from numpy import typing as npt

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

IntegerIndex1D: TypeAlias = Union[int, np.int_, slice, Sequence[int], Sequence[np.int_], npt.NDArray[np.int_]]
BooleanIndex1D: TypeAlias = Union[bool, np.bool_, slice, Sequence[bool], Sequence[np.bool_], npt.NDArray[np.bool_]]
TensorIndex: TypeAlias = Union[IntegerIndex1D, BooleanIndex1D, tuple[IntegerIndex1D, ...], tuple[BooleanIndex1D, ...]]
Shape: TypeAlias = tuple[int, ...]


class NDArrayParameters(TypedDict, total=False):
    dtype: npt.DTypeLike
    copy: bool | None
    order: Literal["K", "A", "C", "F"]
    subok: bool
    ndim: int
    like: npt.ArrayLike


class TensorParameters(NDArrayParameters, total=False):
    covariant: bool | Iterable[int]


class PolytopeParameters(NDArrayParameters, total=False):
    pdim: int


class QuadricParameters(NDArrayParameters, total=False):
    is_dual: bool
    normalize_matrix: bool


NumericalScalarType: TypeAlias = Union[np.number, np.bool_]
NumericalDType: TypeAlias = np.dtype[NumericalScalarType]
NumericalArray: TypeAlias = npt.NDArray[NumericalScalarType]
ScalarNumericalArray: TypeAlias = np.ndarray[tuple[()], NumericalDType]
NumericalScalar: TypeAlias = Union[Number, np.number, np.bool_, ScalarNumericalArray]
