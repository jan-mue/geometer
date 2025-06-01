from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, Literal, TypeVar, cast

import numpy as np
import numpy.typing as npt
from typing_extensions import Self, overload, override

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from geometer.utils.typing import NDArrayParameters, NumericalScalar, TensorParameters

from geometer.base import (
    EQ_TOL_ABS,
    EQ_TOL_REL,
    BoundTensor,
    LeviCivitaTensor,
    ProjectiveTensor,
    Tensor,
    TensorCollection,
    TensorDiagram,
)
from geometer.exceptions import GeometryException, LinearDependenceError, NotCoplanar
from geometer.utils import is_multiple, is_numerical_scalar, matmul, matvec, null_space


# signature when called from meet
@overload
def _join_meet_duality(
    *args: SubspaceTensor,
    intersect_lines: Literal[True] = ...,
    check_dependence: bool = ...,
    normalize_result: bool = ...,
) -> PointTensor | LineTensor: ...


# signature when called from join
@overload
def _join_meet_duality(
    *args: PointTensor | SubspaceTensor,
    intersect_lines: Literal[False],
    check_dependence: bool = ...,
    normalize_result: bool = ...,
) -> SubspaceTensor: ...


def _join_meet_duality(
    *args: PointTensor | SubspaceTensor,
    intersect_lines: bool = True,
    check_dependence: bool = True,
    normalize_result: bool = True,
) -> PointTensor | SubspaceTensor:
    if len(args) < 2:
        raise ValueError(f"Expected at least 2 arguments, got {len(args)}.")

    n = args[0].dim + 1
    result: Tensor

    # all arguments are 1-tensors, i.e. points or hypersurfaces (=lines in 2D)
    if all(o.tensor_shape == args[0].tensor_shape for o in args[1:]) and sum(args[0].tensor_shape) == 1:
        covariant = args[0].tensor_shape[0] > 0
        e = LeviCivitaTensor(n, not covariant)

        # connecting all arguments with e gives a (n-len(args))-tensor (e.g. a 1-tensor for two 2D-lines)
        result = TensorDiagram(*[(o, e) if covariant else (e, o) for o in args]).calculate()

    # two lines/planes
    elif len(args) == 2:
        a, b = args
        if (isinstance(a, LineTensor) and isinstance(b, PlaneTensor)) or (
            isinstance(b, LineTensor) and isinstance(a, PlaneTensor)
        ):
            e = LeviCivitaTensor(n)
            result = TensorDiagram(*[(e, a)] * a.tensor_shape[1], *[(e, b)] * b.tensor_shape[1]).calculate()
        elif isinstance(a, SubspaceTensor) and isinstance(b, PointTensor):
            result = a * b
        elif isinstance(a, PointTensor) and isinstance(b, SubspaceTensor):
            result = b * a
        elif isinstance(a, LineTensor) and isinstance(b, LineTensor):
            # can assume that n >= 4, because for n = 3 lines are 1-tensors
            e = LeviCivitaTensor(n)

            # if this is zero, the lines are coplanar
            result = TensorDiagram(*[(e, a)] * a.tensor_shape[1], *[(e, b)] * (n - a.tensor_shape[1])).calculate()
            coplanar: npt.NDArray[np.bool_] = result.is_zero()

            if np.all(coplanar):
                # this part is inspired by Jim Blinn, Lines in Space: A Tale of Two Lines
                diagram = TensorDiagram(*[(e, a)] * a.tensor_shape[1], (e, b))
                array = diagram.calculate().array

                if np.isscalar(coplanar):
                    i = np.unravel_index(np.abs(array).argmax(), array.shape)
                    if not intersect_lines:
                        # extract the common subspace
                        result = Tensor(array[i[0], ...], covariant=False, copy=None)
                    else:
                        # extract the point of intersection
                        result = Tensor(array[(slice(None),) + i[1:]], copy=None)
                else:
                    max_ind = np.abs(array).reshape((np.prod(array.shape[: coplanar.ndim]), -1)).argmax(1)
                    i = np.unravel_index(max_ind, array.shape[coplanar.ndim :])
                    i = tuple(np.reshape(x, array.shape[: coplanar.ndim]) for x in i)  # type: ignore[misc]
                    indices = tuple(np.indices(array.shape[: coplanar.ndim]))
                    if not intersect_lines:
                        result_array = array[(*indices, i[0], Ellipsis)]  # type: ignore[arg-type]
                        result_rank = result_array.ndim - coplanar.ndim
                        result = Tensor(result_array, covariant=False, tensor_rank=result_rank, copy=None)
                    else:
                        result = Tensor(array[indices + (slice(None),) + i[1:]], tensor_rank=1, copy=None)

            elif intersect_lines or n == 4:
                # can't intersect lines that are not coplanar and can't join skew lines in 3D
                raise NotCoplanar("The given lines are not all coplanar.")
            elif np.any(coplanar) and (a.free_indices > 0 or b.free_indices > 0):
                raise GeometryException("Can only join tensors that are either all coplanar or all not coplanar.")

        else:
            raise ValueError(f"Arguments of type {type(a)} and {type(b)} are not supported.")

    else:
        raise ValueError(
            f"Expected all arguments to be 1-tensors or a pair of lines/planes, but got {len(args)} tensors of higher rank."
        )

    if check_dependence:
        is_zero = result.is_zero()
        if result.free_indices == 0 and is_zero:
            raise LinearDependenceError("Arguments are not linearly independent.")
        elif np.any(is_zero):
            raise LinearDependenceError("Some arguments are not linearly independent.", is_zero)

    if normalize_result:
        axes = tuple(result._covariant_indices) + tuple(result._contravariant_indices)
        max_abs = np.max(np.abs(result.array), axis=axes, keepdims=True)
        _, max_exponent = np.frexp(max_abs)
        result.array = _divide_by_power_of_two(result.array, max_exponent)

    if result.tensor_shape == (0, 1):
        return LineCollection.from_tensor(result) if n == 3 else PlaneCollection.from_tensor(result)
    if result.tensor_shape == (1, 0):
        return PointCollection.from_tensor(result)
    if result.tensor_shape == (2, 0):
        return LineCollection.from_tensor(result).contravariant_tensor
    if result.tensor_shape == (0, n - 2):
        return LineCollection.from_tensor(result)

    raise RuntimeError(f"Unexpected tensor of type {result.tensor_shape}")


def _divide_by_power_of_two(array: np.ndarray, power: int) -> np.ndarray:
    if array.dtype.kind == "c":
        rm, re = np.frexp(array.real)
        im, ie = np.frexp(array.imag)
        re -= power
        ie -= power
        out = np.empty_like(array)
        np.ldexp(rm, re, out=out.real)
        np.ldexp(im, ie, out=out.imag)
        return out

    mantissa, exponent = np.frexp(array)
    exponent -= power
    return np.ldexp(mantissa, exponent)


@overload
def join(*args: Unpack[tuple[Point, Point]], _check_dependence: bool = ..., _normalize_result: bool = ...) -> Line: ...


@overload
def join(
    *args: Unpack[tuple[PointCollection, PointTensor]], _check_dependence: bool = ..., _normalize_result: bool = ...
) -> LineCollection: ...


@overload
def join(
    *args: Unpack[tuple[PointTensor, PointCollection]], _check_dependence: bool = ..., _normalize_result: bool = ...
) -> LineCollection: ...


@overload
def join(
    *args: Unpack[tuple[PointTensor, PointTensor]], _check_dependence: bool = ..., _normalize_result: bool = ...
) -> LineTensor: ...


@overload
def join(
    *args: Unpack[tuple[Point, Point, Point]],
    _check_dependence: bool = ...,
    _normalize_result: bool = ...,
) -> Plane: ...


@overload
def join(
    *args: Unpack[tuple[PointCollection, PointTensor, PointTensor]],
    _check_dependence: bool = ...,
    _normalize_result: bool = ...,
) -> PlaneCollection: ...


@overload
def join(
    *args: Unpack[tuple[PointTensor, PointCollection, PointTensor]],
    _check_dependence: bool = ...,
    _normalize_result: bool = ...,
) -> PlaneCollection: ...


@overload
def join(
    *args: Unpack[tuple[PointTensor, PointTensor, PointCollection]],
    _check_dependence: bool = ...,
    _normalize_result: bool = ...,
) -> PlaneCollection: ...


@overload
def join(
    *args: Unpack[tuple[Point, Line]],
    _check_dependence: bool = ...,
    _normalize_result: bool = ...,
) -> Plane: ...


@overload
def join(
    *args: Unpack[tuple[Line, Point]],
    _check_dependence: bool = ...,
    _normalize_result: bool = ...,
) -> Plane: ...


@overload
def join(
    *args: Unpack[tuple[PointCollection, LineTensor]],
    _check_dependence: bool = ...,
    _normalize_result: bool = ...,
) -> PlaneCollection: ...


@overload
def join(
    *args: Unpack[tuple[LineTensor, PointCollection]],
    _check_dependence: bool = ...,
    _normalize_result: bool = ...,
) -> PlaneCollection: ...


@overload
def join(
    *args: Unpack[tuple[PointTensor, LineCollection]],
    _check_dependence: bool = ...,
    _normalize_result: bool = ...,
) -> PlaneCollection: ...


@overload
def join(
    *args: Unpack[tuple[LineCollection, PointTensor]],
    _check_dependence: bool = ...,
    _normalize_result: bool = ...,
) -> PlaneCollection: ...


@overload
def join(
    *args: Unpack[tuple[LineTensor, PointTensor | LineTensor]],
    _check_dependence: bool = ...,
    _normalize_result: bool = ...,
) -> PlaneTensor: ...


@overload
def join(
    *args: Unpack[tuple[PointTensor | LineTensor, LineTensor]],
    _check_dependence: bool = ...,
    _normalize_result: bool = ...,
) -> PlaneTensor: ...


@overload
def join(
    *args: PointTensor | LineTensor, _check_dependence: bool = ..., _normalize_result: bool = ...
) -> SubspaceTensor: ...


def join(
    *args: PointTensor | LineTensor, _check_dependence: bool = True, _normalize_result: bool = True
) -> SubspaceTensor:
    """Joins a number of objects to form a line or plane.

    Args:
        *args: Objects to join, e.g. 2 points, lines, a point and a line or 3 points.

    Returns:
        The resulting line or plane.

    Raises:
        LinearDependenceError: If two objects coincide.
        NotCoplanar: For two skew lines in 3D.

    """
    return _join_meet_duality(
        *args, intersect_lines=False, check_dependence=_check_dependence, normalize_result=_normalize_result
    )


@overload
def meet(*args: LineTensor, _check_dependence: bool = ..., _normalize_result: bool = ...) -> PointTensor: ...


@overload
def meet(
    *args: Unpack[tuple[SubspaceTensor, LineTensor]], _check_dependence: bool = ..., _normalize_result: bool = ...
) -> PointTensor: ...


@overload
def meet(
    *args: Unpack[tuple[LineTensor, SubspaceTensor]], _check_dependence: bool = ..., _normalize_result: bool = ...
) -> PointTensor: ...


@overload
def meet(
    *args: Unpack[tuple[PlaneTensor, PlaneTensor]], _check_dependence: bool = ..., _normalize_result: bool = ...
) -> LineTensor: ...


# TODO: add overloads for non-collection and collection subtypes


@overload
def meet(
    *args: SubspaceTensor, _check_dependence: bool = ..., _normalize_result: bool = ...
) -> PointTensor | LineTensor: ...


def meet(
    *args: SubspaceTensor, _check_dependence: bool = True, _normalize_result: bool = True
) -> PointTensor | LineTensor:
    """Intersects a number of given objects.

    Args:
        *args: Objects to intersect, e.g. two lines, planes, a plane and a line or 3 planes.

    Returns:
        The resulting point, line or subspace.

    Raises:
        LinearDependenceError: If two subspaces coincide.
        NotCoplanar: If two lines are not coplanar.

    """
    return _join_meet_duality(
        *args, intersect_lines=True, check_dependence=_check_dependence, normalize_result=_normalize_result
    )


class PointLikeTensor(ProjectiveTensor, ABC):
    """Base class for point tensors and polytopes implementing arithmetic operations."""

    def __init__(
        self,
        *args: Tensor | npt.ArrayLike,
        homogenize: bool = False,
        tensor_rank: int = 1,
        **kwargs: Unpack[NDArrayParameters],
    ) -> None:
        super().__init__(*args, tensor_rank=tensor_rank, **kwargs)
        if homogenize is True:
            self.array = np.append(self.array, np.ones(self.shape[:-1] + (1,), self.dtype), axis=-1)
        if self.tensor_shape != (1, 0):
            raise ValueError(f"Expected tensor of type (1, 0), but is {self.tensor_shape}")

    @staticmethod
    def _normalize_array(array: np.ndarray) -> np.ndarray:
        z = array[..., -1, None]
        isinf = np.isclose(z, 0, atol=EQ_TOL_ABS)
        if np.all(isinf | (z == 1)):
            return array
        dtype = np.promote_types(np.float64, array.dtype)
        result = array.astype(dtype)
        np.divide(result, z, where=~isinf, out=result)
        return np.real_if_close(result)

    @property
    def normalized_array(self) -> np.ndarray:
        """The coordinate array of the points with the last coordinates normalized to 1."""
        return self._normalize_array(self.array)

    @overload
    def __add__(self, other: PointLikeTensor) -> PointCollection | Point: ...

    @overload
    def __add__(self, other: Tensor | npt.ArrayLike) -> Tensor: ...

    @override
    def __add__(self, other: Tensor | npt.ArrayLike) -> Tensor:
        if not isinstance(other, PointLikeTensor):
            return super().__add__(other)
        a, b = self.normalized_array, other.normalized_array
        result = a[..., :-1] + b[..., :-1]
        result = np.append(result, np.maximum(a[..., -1:], b[..., -1:]), axis=-1)
        return PointCollection.from_array(result)

    @overload
    def __sub__(self, other: PointLikeTensor) -> PointCollection | Point: ...

    @overload
    def __sub__(self, other: Tensor | npt.ArrayLike) -> Tensor: ...

    @override
    def __sub__(self, other: Tensor | npt.ArrayLike) -> Tensor:
        if not isinstance(other, PointLikeTensor):
            return super().__sub__(other)
        a, b = self.normalized_array, other.normalized_array
        result = a[..., :-1] - b[..., :-1]
        result = np.append(result, np.maximum(a[..., -1:], b[..., -1:]), axis=-1)
        return PointCollection.from_array(result)

    @overload
    def __mul__(self, other: NumericalScalar) -> PointCollection | Point: ...

    @overload
    def __mul__(self, other: Tensor | npt.ArrayLike) -> Tensor: ...

    @override  # type: ignore[misc]
    def __mul__(self, other: Tensor | npt.ArrayLike) -> Tensor:
        if not is_numerical_scalar(other):
            return super().__mul__(other)
        result = self.normalized_array[..., :-1] * other
        result = np.append(result, self.array[..., -1:] != 0, axis=-1)
        return PointCollection.from_array(result)

    @overload
    def __truediv__(self, other: NumericalScalar) -> PointCollection | Point: ...

    @overload
    def __truediv__(self, other: Tensor | npt.ArrayLike) -> Tensor: ...

    @override  # type: ignore[misc]
    def __truediv__(self, other: Tensor | npt.ArrayLike) -> Tensor:
        if not is_numerical_scalar(other):
            return super().__truediv__(other)
        result = self.normalized_array[..., :-1] / other
        result = np.append(result, self.array[..., -1:] != 0, axis=-1)
        return PointCollection.from_array(result)

    @override
    def __neg__(self) -> PointCollection | Point:
        result = -self.normalized_array[..., :-1]
        result = np.append(result, self.array[..., -1:] != 0, axis=-1)
        return PointCollection.from_array(result)


class PointTensor(PointLikeTensor, ABC):
    """Represents points in a projective space of arbitrary dimension.

    The number of supplied coordinates determines the dimension of the space that the point lives in.
    If the coordinates are given as scalar arguments (not in a single iterable), the coordinates will automatically be
    transformed into homogeneous coordinates, i.e. a one added as an additional coordinate.

    Addition and subtraction of finite and infinite points will always give a finite result if one of the points
    was finite beforehand.

    Args:
        *args: A single iterable object or tensor or multiple (affine) coordinates.
        homogenize: If True, and the first argument is an array of points, all points in the array will be converted to
            homogeneous coordinates, i.e. 1 will be added to the coordinates of each point.
            By default, homogenize is False.
        **kwargs: Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    """

    def __init__(
        self,
        *args: Tensor | npt.ArrayLike,
        homogenize: bool = False,
        tensor_rank: int = 1,
        **kwargs: Unpack[NDArrayParameters],
    ) -> None:
        if np.isscalar(args[0]):
            super().__init__(*args, 1, homogenize=False, tensor_rank=tensor_rank, **kwargs)
        else:
            super().__init__(*args, homogenize=homogenize, tensor_rank=tensor_rank, **kwargs)

    @override
    def __repr__(self) -> str:
        if self.free_indices == 0:
            result = f"Point({', '.join(self.normalized_array[:-1].astype(str))})"
            if self.isinf:
                result += " at Infinity"
            return result
        return f"PointCollection({self.normalized_array.tolist()})"

    @overload
    def join(self, *others: Unpack[tuple[PointTensor]]) -> LineTensor: ...

    @overload
    def join(self, *others: Unpack[tuple[PointTensor, PointTensor]]) -> PlaneTensor: ...

    @overload
    def join(self, *others: LineTensor) -> PlaneTensor: ...

    @overload
    def join(self, *others: PointTensor) -> SubspaceTensor: ...

    def join(self, *others: PointTensor | LineTensor) -> SubspaceTensor:
        return join(self, *others)

    def _matrix_transform(self, m: npt.ArrayLike) -> PointCollection | Point:
        if self.free_indices == 0:
            return PointCollection.from_array(np.dot(m, self.array))
        return PointCollection.from_array(matvec(m, self.array))

    @property
    def isinf(self) -> npt.NDArray[np.bool_]:
        """Boolean array that indicates which points lie at infinity."""
        return np.isclose(self.array[..., -1], 0, atol=EQ_TOL_ABS)

    @property
    def isreal(self) -> npt.NDArray[np.bool_]:
        """Boolean array that indicates which points are real."""
        return np.all(np.isreal(self.normalized_array), axis=-1)


class Point(PointTensor, BoundTensor):
    @overload
    def join(self, *others: Point) -> Line: ...

    @overload
    def join(self, *others: Line) -> Plane: ...

    @overload
    def join(self, *others: PointCollection) -> LineCollection: ...

    @overload
    def join(self, *others: LineCollection) -> PlaneCollection: ...

    @overload
    def join(self, *others: PointTensor | LineTensor) -> SubspaceTensor: ...

    @override
    def join(self, *others: PointTensor | LineTensor) -> SubspaceTensor:
        return join(self, *others)

    @overload
    def __mul__(self, other: NumericalScalar) -> Point: ...

    @overload
    def __mul__(self, other: Tensor | npt.ArrayLike) -> Tensor: ...

    @override  # type: ignore[misc]
    def __mul__(self, other: Tensor | npt.ArrayLike) -> Tensor:
        return super().__mul__(other)

    @overload
    def __truediv__(self, other: NumericalScalar) -> Point: ...

    @overload
    def __truediv__(self, other: Tensor | npt.ArrayLike) -> Tensor: ...

    @override  # type: ignore[misc]
    def __truediv__(self, other: Tensor | npt.ArrayLike) -> Tensor:
        return super().__truediv__(other)

    @override
    def __neg__(self) -> Point:
        return cast(Point, super().__neg__())


class PointCollection(PointTensor, TensorCollection[Point]):
    _element_class = Point

    @overload
    def join(self, *others: PointTensor) -> LineCollection: ...

    @overload
    def join(self, *others: LineTensor) -> PlaneCollection: ...

    @override
    def join(self, *others: PointTensor | LineTensor) -> SubspaceTensor:
        return join(self, *others)


class SubspaceTensor(ProjectiveTensor, ABC):
    """Abstract base class for subspaces of a projective space. Line and Plane are subclasses.

    Args:
        *args: The coordinates of the subspace. Instead of separate coordinates, a single iterable can be supplied.
        tensor_rank: The rank of the tensors that represent the subspace(s). Default is 1.
        **kwargs: Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    """

    def __init__(self, *args: Tensor | npt.ArrayLike, tensor_rank: int = 1, **kwargs: Unpack[TensorParameters]) -> None:
        kwargs.setdefault("covariant", False)
        super().__init__(*args, tensor_rank=tensor_rank, **kwargs)

    @overload
    def meet(self, other: LineCollection) -> PointCollection: ...

    @overload
    def meet(self, other: LineTensor) -> PointTensor: ...

    @overload
    def meet(self, other: SubspaceTensor) -> PointTensor | LineTensor: ...

    def meet(self, other: SubspaceTensor) -> PointTensor | LineTensor:
        return meet(self, other)

    @overload
    def __add__(self, other: PointTensor) -> Self: ...

    @overload
    def __add__(self, other: Tensor | npt.ArrayLike) -> Tensor: ...

    @override
    def __add__(self, other: Tensor | npt.ArrayLike) -> Tensor:
        if not isinstance(other, PointTensor):
            return super().__add__(other)

        from geometer.transformation import translation

        return translation(other).apply(self)

    @overload
    def __sub__(self, other: PointTensor) -> Self: ...

    @overload
    def __sub__(self, other: Tensor | npt.ArrayLike) -> Tensor: ...

    @override
    def __sub__(self, other: Tensor | npt.ArrayLike) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other, copy=None)
        return self + (-other)

    @property
    def basis_matrix(self) -> npt.NDArray[np.number]:
        x = self.array
        x = x.reshape(x.shape[: self.free_indices] + (-1, x.shape[-1]))
        return np.swapaxes(null_space(x, self.shape[-1] - self.rank + self.free_indices), -1, -2)

    def _matrix_transform(self, m: npt.ArrayLike) -> SubspaceTensor:
        transformed_basis = matmul(self.basis_matrix, m, transpose_b=True)
        transformed_basis_points = [PointCollection.from_array(p) for p in np.swapaxes(transformed_basis, 0, -2)]
        return join(*transformed_basis_points)

    @property
    def general_point(self) -> PointTensor:
        """Point in general position i.e. not in the subspaces."""
        n = self.dim + 1
        s = [self.shape[i] for i in range(self.free_indices)]
        p = PointCollection.from_array(np.zeros([*s, n], dtype=int))
        ind = np.ones(s, dtype=bool)
        for i in range(n):
            p[ind, -i - 1] = 1
            ind = self.contains(p)
            if not np.any(ind):
                break
        return p

    @overload
    def parallel(self, through: Point) -> Self: ...

    @overload
    def parallel(self, through: PointTensor) -> SubspaceTensor: ...

    def parallel(self, through: PointTensor) -> SubspaceTensor:
        """Returns the subspace through given point that is parallel to this subspace.

        Args:
            through: The point through which the parallel subspace is to be constructed.

        Returns:
            The parallel subspace.

        """
        x = self.meet(infty_hyperplane(self.dim))
        return join(x, through)

    def contains(self, other: PointTensor | LineTensor, tol: float = EQ_TOL_ABS) -> npt.NDArray[np.bool_]:
        """Tests whether given points or lines lie in the subspaces.

        Args:
            other: The object(s) to test.
            tol: The accepted tolerance.

        Returns:
            Boolean array that indicates which of given points/lines lies in the subspaces.

        """
        if isinstance(other, PointTensor):
            result = self * other
        elif isinstance(other, LineTensor):
            result = self * other.covariant_tensor
        else:
            raise TypeError(f"argument of type {type(other)} not supported")

        axes = tuple(result._covariant_indices) + tuple(result._contravariant_indices)
        return np.all(np.isclose(result.array, 0, atol=tol), axis=axes)

    def is_parallel(self, other: LineTensor | PlaneTensor) -> npt.NDArray[np.bool_]:
        """Tests whether the given subspace is parallel to this subspace.

        Args:
            other: The other space to test.

        Returns:
            True, if the two subspaces are parallel.

        """
        x = self.meet(other)
        return infty_hyperplane(self.dim).contains(x)

    @abstractmethod
    def mirror(self, pt: PointTensor) -> PointTensor:
        """Construct the reflection of a point at the subspace.

        Args:
            pt: The point to reflect.

        Returns:
            The mirror point.

        """

    @abstractmethod
    def perpendicular(self, through: PointTensor) -> LineTensor:
        """Construct the perpendicular subspace though the given point or line.

        Args:
            through: The point or line through which the perpendicular is constructed.

        Returns:
            The perpendicular subspace.

        """

    def project(self, pt: PointTensor) -> PointTensor:
        """The orthogonal projection of a point onto the subspace.

        Args:
            pt: The point to project.

        Returns:
            The projected point.

        """
        l = self.perpendicular(pt)
        return self.meet(l)


class Subspace(SubspaceTensor, BoundTensor, ABC):
    @overload
    def meet(self, other: Line) -> Point: ...

    @overload
    def meet(self, other: Subspace) -> Point | Line: ...

    @overload
    def meet(self, other: LineCollection) -> PointCollection: ...

    @overload
    def meet(self, other: SubspaceCollection[SubspaceT]) -> PointCollection | LineCollection: ...

    @overload
    def meet(self, other: SubspaceTensor) -> PointTensor | LineTensor: ...

    @override
    def meet(self, other: SubspaceTensor) -> PointTensor | LineTensor:
        return super().meet(other)


SubspaceT = TypeVar("SubspaceT", covariant=True, bound=Subspace)


class SubspaceCollection(SubspaceTensor, TensorCollection[SubspaceT], Generic[SubspaceT], ABC):
    pass


class LineTensor(SubspaceTensor, ABC):
    """Represents a line in a projective space of arbitrary dimension.

    Args:
        *args: Two points or the coordinates of the line. Instead of all coordinates separately, a single iterable can
            also be supplied.
        **kwargs: Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    """

    def __init__(self, *args: Tensor | npt.ArrayLike, **kwargs: Unpack[NDArrayParameters]) -> None:
        if len(args) == 2:
            kwargs["copy"] = False
            p, q = args
            super().__init__(
                join(PointCollection.from_array(p), PointCollection.from_array(q)), tensor_rank=-2, **kwargs
            )
        else:
            super().__init__(*args, tensor_rank=-2, **kwargs)
        if self.tensor_shape not in {(0, self.dim - 1), (self.dim - 1, 0)}:
            raise ValueError(f"Unexpected tensor of type {self.tensor_shape}")
        if self.dim == 3 and self.shape[-1] != self.shape[-2]:
            raise ValueError(f"Expected quadratic matrix, but last two dimensions are {self.shape[-2:]}")

    @override
    def _matrix_transform(self, m: npt.ArrayLike) -> LineTensor:
        result = super()._matrix_transform(m)
        return cast(LineTensor, result)

    @overload
    def meet(self, other: SubspaceCollection[SubspaceT]) -> PointCollection: ...

    @overload
    def meet(self, other: SubspaceTensor) -> PointTensor: ...

    @override
    def meet(self, other: SubspaceTensor) -> PointTensor:
        result = super().meet(other)
        return cast(PointTensor, result)

    def join(self, *others: PointTensor | LineTensor) -> PlaneTensor:
        return join(self, *others)

    @property
    def base_point(self) -> PointTensor:
        """Base points for the lines, arbitrarily chosen."""
        if self.dim > 2:
            base = self.basis_matrix
            p, q = base[..., 0, :], base[..., 1, :]
            isinf = np.isclose(p[..., -1, None], 0, atol=EQ_TOL_ABS)
            result = np.where(isinf, q, p)
            return PointCollection.from_array(result)

        y_zero = np.isclose(self.array[..., 1], 0, atol=EQ_TOL_ABS)
        z_zero = np.isclose(self.array[..., 2], 0, atol=EQ_TOL_ABS)

        result = np.zeros_like(self.array)
        result[z_zero, 2] = 1

        result[~y_zero & ~z_zero, 1] = -self.array[~y_zero & ~z_zero, 2]
        result[~y_zero & ~z_zero, 2] = self.array[~y_zero & ~z_zero, 1]

        result[y_zero & ~z_zero, 0] = self.array[y_zero & ~z_zero, 2]
        result[y_zero & ~z_zero, 2] = -self.array[y_zero & ~z_zero, 0]

        return PointCollection.from_array(result)

    @property
    def direction(self) -> PointTensor:
        """The direction of the lines (not normalized)."""
        if self.dim > 2:
            return meet(self, infty_hyperplane(self.dim), _normalize_result=False)

        x_zero = np.isclose(self.array[..., 0], 0, atol=EQ_TOL_ABS)
        y_zero = np.isclose(self.array[..., 1], 0, atol=EQ_TOL_ABS)

        result = np.zeros_like(self.array)
        result[x_zero & y_zero, 1] = 1

        result[~(x_zero & y_zero), 0] = self.array[~(x_zero & y_zero), 1]
        result[~(x_zero & y_zero), 1] = -self.array[~(x_zero & y_zero), 0]

        return PointCollection.from_array(result)

    @override
    @property
    def basis_matrix(self) -> np.ndarray:
        """A matrix with orthonormal basis vectors as rows."""
        if self.dim == 2:
            a = self.base_point.array
            b = np.cross(self.array, a)  # type: ignore[arg-type]
            m = [a / np.linalg.norm(a, axis=-1, keepdims=True), b / np.linalg.norm(b, axis=-1, keepdims=True)]
            return np.stack(m, axis=-2)
        return super().basis_matrix

    @property
    def covariant_tensor(self) -> LineTensor:
        """The covariant tensors of lines in 3D."""
        if self.dim != 3:
            raise NotImplementedError(f"Expected dimension 3 but found dimension {self.dim}.")
        if self.tensor_shape[0] > 0:
            return self
        e = LeviCivitaTensor(4)
        diagram = TensorDiagram((e, self), (e, self))
        return LineCollection.from_tensor(diagram.calculate())

    @property
    def contravariant_tensor(self) -> LineTensor:
        """The contravariant tensors of lines in 3D."""
        if self.dim != 3:
            raise NotImplementedError(f"Expected dimension 3 but found dimension {self.dim}.")
        if self.tensor_shape[1] > 0:
            return self
        e = LeviCivitaTensor(4, False)
        diagram = TensorDiagram((self, e), (self, e))
        return LineCollection.from_tensor(diagram.calculate())

    def is_coplanar(self, other: LineTensor) -> npt.NDArray[np.bool_]:
        """Tests whether another line lies in the same plane as this line, i.e. whether two lines intersect.

        Args:
            other: A line in 3D to test.

        Returns:
            True if the two lines intersect (i.e. they lie in the same plane).

        References:
          - Jim Blinn, Lines in Space: Back to the Diagrams, Line Intersections

        """
        if self.dim == 2:
            return np.ones(self.shape[: self.free_indices], dtype=bool)

        e = LeviCivitaTensor(self.dim + 1)
        d = TensorDiagram(*[(e, self)] * (self.dim - 1), *[(e, other)] * (self.dim - 1))
        return d.calculate().is_zero()

    @override
    def mirror(self, pt: PointTensor) -> PointTensor:
        """Construct the reflection of points at the lines.

        Args:
            pt: The point to reflect.

        Returns:
            The mirror points.

        References:
          - J. Richter-Gebert: Perspectives on Projective Geometry, Section 19.1

        """
        l = self
        if self.dim >= 3:
            # TODO: handle points on self
            e = join(self, pt)

            if self.dim == 3:
                f = e.perpendicular(self)
                return f.mirror(pt)

            m = e.basis_matrix
            arr = matvec(m, pt.array)
            arr_sort = np.argsort(np.abs(arr), axis=-1)
            arr_ind = tuple(np.indices(arr.shape)[:-1])
            m = m[(*arr_ind, arr_sort, slice(None))]  # type: ignore[arg-type]
            pt = PointCollection(arr[(*arr_ind, arr_sort)], copy=None)
            l = self._matrix_transform(m)
        l1 = join(I, pt, _normalize_result=False)
        l2 = join(J, pt, _normalize_result=False)
        p1 = l.meet(l1)
        p2 = l.meet(l2)
        m1 = join(p1, J, _normalize_result=False)
        m2 = join(p2, I, _normalize_result=False)
        result = m1.meet(m2)
        if self.dim >= 3:
            return result._matrix_transform(np.swapaxes(m, -1, -2))
        return result

    @override
    def perpendicular(self, through: PointTensor, plane: PlaneTensor | None = None) -> LineTensor:
        """Construct the perpendicular line though a point.

        Args:
            through: The point through which the perpendicular is constructed.
            plane: In three-dimensional spaces, the 2-dimensional subspace that the perpendicular line is
                supposed to lie in, can be specified.

        Returns:
            The perpendicular line.

        """
        n = self.dim + 1
        contains = self.contains(through)
        result = LineCollection.from_array(np.empty(contains.shape + (n,) * (n - 2), np.complex128))
        if np.any(contains):
            l = self
            if self.free_indices > 0:
                l = self[contains]  # type: ignore[assignment]

            if n > 3:
                if plane is None:
                    # additional point is required to determine the exact line
                    plane = join(l, l.general_point)
                elif isinstance(plane, PlaneCollection):
                    plane = plane[contains]  # type: ignore[assignment]

                basis = plane.basis_matrix
                line_pts = matmul(l.basis_matrix, basis, transpose_b=True)
                l = LineCollection.from_array(np.cross(line_pts[..., 0, :], line_pts[..., 1, :]))  # type: ignore[arg-type]

            p = PointCollection.from_array(
                np.append(l.array[..., :-1], np.zeros(l.shape[:-1] + (1,), dtype=l.dtype), axis=-1)
            )

            if n > 3:
                p = p._matrix_transform(np.swapaxes(basis, -1, -2))

            result[contains] = join(through if through.free_indices == 0 else through[contains], p)  # type: ignore[call-arg, arg-type]

        if np.any(~contains):
            if through.free_indices > 0:
                through = through[~contains]  # type: ignore[assignment]
            if self.free_indices > 0:
                result[~contains] = cast(LineTensor, self[~contains]).mirror(through).join(through)
            else:
                result[~contains] = self.mirror(through).join(through)

        return LineCollection.from_array(np.real_if_close(result.array))


class Line(LineTensor, Subspace):
    @overload
    def meet(self, other: Subspace) -> Point: ...

    @overload
    def meet(self, other: SubspaceCollection[SubspaceT]) -> PointCollection: ...

    @overload
    def meet(self, other: SubspaceTensor) -> PointTensor: ...

    @override
    def meet(self, other: SubspaceTensor) -> PointTensor:
        return super().meet(other)

    @overload
    def join(self, *others: Point | Line) -> Plane: ...

    @overload
    def join(self, *others: PointCollection | LineCollection) -> PlaneCollection: ...

    @overload
    def join(self, *others: PointTensor | LineTensor) -> PlaneTensor: ...

    @override
    def join(self, *others: PointTensor | LineTensor) -> PlaneTensor:
        return super().join(*others)


class LineCollection(LineTensor, SubspaceCollection[Line]):
    _element_class = Line


class PlaneTensor(SubspaceTensor):
    """Represents a hyperplane in a projective space of arbitrary dimension.

    Args:
        *args: The points/lines spanning the plane or the coordinates of the hyperplane. Instead of separate
            coordinates, a single iterable can be supplied.
        **kwargs: Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    """

    def __init__(self, *args: Tensor | npt.ArrayLike, **kwargs: Unpack[NDArrayParameters]) -> None:
        if all(isinstance(o, (LineTensor, PointTensor)) for o in args):
            kwargs["copy"] = False
            super().__init__(join(*args), **kwargs)  # type: ignore[arg-type, call-arg]
        else:
            super().__init__(*args, **kwargs)
        if self.tensor_shape != (0, 1):
            raise ValueError(f"Expected tensor of type (0, 1), but is {self.tensor_shape}")

    @overload
    def meet(self, other: LineTensor) -> PointTensor: ...

    @overload
    def meet(self, other: PlaneTensor) -> LineTensor: ...

    @overload
    def meet(self, other: SubspaceTensor) -> PointTensor | LineTensor: ...

    @override
    def meet(self, other: SubspaceTensor) -> PointTensor | LineTensor:
        return super().meet(other)

    @override
    @property
    def basis_matrix(self) -> np.ndarray:
        n = self.dim + 1
        i = np.argmax(np.abs(self.array) > EQ_TOL_ABS, axis=-1, keepdims=True)
        result = np.zeros(self.shape[:-1] + (n, n - 1), dtype=self.dtype)
        ind = np.indices(self.shape)
        ind_short = np.indices(self.shape[:-1], sparse=True)
        s = ind.shape
        a = tuple(np.delete(ind, np.ravel_multi_index((*tuple(np.indices(s)[:-1]), i), s)).reshape(s[:-1] + (-1,)))
        result[(*ind_short, i.squeeze())] = self.array[a]
        b = np.arange(n - 1).reshape((1,) * (len(self.shape) - 1) + (n - 1,))
        result[(*a, b)] = -self.array[(*ind_short, i.squeeze(), None)]  # type: ignore[misc, index]
        q, r = np.linalg.qr(result)  # type: ignore[arg-type]
        return np.swapaxes(q, -1, -2)

    @override
    def mirror(self, pt: PointTensor) -> PointTensor:
        """Construct the reflections of a point at the plane.

        Only works in 3D.

        Args:
            pt: The points to reflect.

        Returns:
            The mirror points.

        """
        if self.dim != 3:
            raise NotImplementedError(f"Expected dimension 3 but found dimension {self.dim}.")
        l = self.meet(infty_plane)
        basis = l.basis_matrix
        l = LineCollection.from_array(np.cross(basis[..., 0, :-1], basis[..., 1, :-1]))
        p = l.base_point
        polars = LineCollection.from_array(p.array)

        from geometer.curve import absolute_conic

        p1, p2 = absolute_conic.intersect(polars)
        p1 = PointCollection.from_array(np.append(p1.array, np.zeros(p1.shape[:-1] + (1,)), axis=-1))
        p2 = PointCollection.from_array(np.append(p2.array, np.zeros(p2.shape[:-1] + (1,)), axis=-1))

        l1 = p1.join(pt)
        l2 = p2.join(pt)
        q1 = self.meet(l1)
        q2 = self.meet(l2)
        m1 = q1.join(p2)
        m2 = q2.join(p1)
        return m1.meet(m2)  # type: ignore[return-value]

    @overload
    def perpendicular(self, through: PointTensor) -> LineTensor: ...

    @overload
    def perpendicular(self, through: LineTensor) -> PlaneTensor: ...

    @override
    def perpendicular(self, through: PointTensor | LineTensor) -> LineTensor | PlaneTensor:
        """Construct the perpendicular lines though the given points or the perpendicular planes through the given lines.

        Only works for lines in 3D.

        Args:
            through: The points or lines through which the perpendiculars are constructed.

        Returns:
            The perpendicular lines or planes.

        """
        if self.dim != 3 and isinstance(through, LineTensor):
            raise NotImplementedError(f"Expected dimension 3 but found dimension {self.dim}.")

        direction_array = self.array[..., :-1]
        direction_array = np.append(direction_array, np.zeros(self.shape[:-1] + (1,), dtype=self.dtype), axis=-1)
        direction = PointCollection.from_array(direction_array)
        return through.join(direction)

    @property
    def isinf(self) -> np.bool_ | npt.NDArray[np.bool_]:
        """Boolean array that indicates whether the plane is the hyperplane at infinity."""
        return is_multiple(self.array, infty_hyperplane(self.dim).array, axis=-1, rtol=EQ_TOL_REL, atol=EQ_TOL_ABS)


class Plane(PlaneTensor, Subspace):
    @overload
    def meet(self, other: Line) -> Point: ...

    @overload
    def meet(self, other: Plane) -> Line: ...

    @overload
    def meet(self, other: LineCollection) -> PointCollection: ...

    @overload
    def meet(self, other: PlaneCollection) -> LineCollection: ...

    @overload
    def meet(self, other: LineTensor) -> PointTensor: ...

    @overload
    def meet(self, other: PlaneTensor) -> LineTensor: ...

    @overload
    def meet(self, other: SubspaceTensor) -> PointTensor | LineTensor: ...

    @override
    def meet(self, other: SubspaceTensor) -> PointTensor | LineTensor:
        return super().meet(other)


class PlaneCollection(PlaneTensor, SubspaceCollection[Plane]):
    _element_class = Plane


I = Point([-1j, 1, 0])
J = Point([1j, 1, 0])
infty = Line(0, 0, 1)


@overload
def infty_hyperplane(dimension: Literal[2]) -> Line: ...


@overload
def infty_hyperplane(dimension: Literal[3]) -> Plane: ...


@overload
def infty_hyperplane(dimension: int) -> Line | Plane: ...


def infty_hyperplane(dimension: int) -> Line | Plane:
    if dimension == 2:
        return infty
    return Plane([0] * dimension + [1])


infty_plane = infty_hyperplane(3)
