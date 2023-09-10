from __future__ import annotations

from typing import TYPE_CHECKING, cast, overload

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from typing_extensions import Literal, Unpack

from geometer.base import EQ_TOL_ABS, LeviCivitaTensor, ProjectiveTensor, Tensor, TensorDiagram, TensorIndex
from geometer.exceptions import GeometryException, LinearDependenceError, NotCoplanar
from geometer.utils import matmul, matvec, null_space


@overload
def _join_meet_duality(
    *args: Line, intersect_lines: Literal[True], check_dependence: bool = True, normalize_result: bool = True
) -> Point:
    ...


@overload
def _join_meet_duality(
    *args: Unpack[tuple[Subspace, Line]],
    intersect_lines: Literal[True],
    check_dependence: bool = True,
    normalize_result: bool = True,
) -> Point:
    ...


@overload
def _join_meet_duality(
    *args: Unpack[tuple[Line, Subspace]],
    intersect_lines: Literal[True],
    check_dependence: bool = True,
    normalize_result: bool = True,
) -> Point:
    ...


@overload
def _join_meet_duality(
    *args: Unpack[tuple[Point, Point]],
    intersect_lines: Literal[True],
    check_dependence: bool = True,
    normalize_result: bool = True,
) -> Line:
    ...


@overload
def _join_meet_duality(
    *args: Point | Subspace,
    intersect_lines: Literal[False],
    check_dependence: bool = True,
    normalize_result: bool = True,
) -> Subspace:
    ...


def _join_meet_duality(
    *args: Point | Subspace, intersect_lines: bool = True, check_dependence: bool = True, normalize_result: bool = True
) -> Point | Subspace:
    if len(args) < 2:
        raise ValueError(f"Expected at least 2 arguments, got {len(args)}.")

    n = args[0].dim + 1

    # all arguments are 1-tensors, i.e. points or hypersurfaces (=lines in 2D)
    if all(o.tensor_shape == args[0].tensor_shape for o in args[1:]) and sum(args[0].tensor_shape) == 1:
        covariant = args[0].tensor_shape[0] > 0
        e = LeviCivitaTensor(n, not covariant)

        # summing all arguments with e gives a (n-len(args))-tensor (e.g. a 1-tensor for two 2D-lines)
        result = TensorDiagram(*[(o, e) if covariant else (e, o) for o in args]).calculate()

    # two lines/planes
    elif len(args) == 2:
        a, b = args
        if isinstance(a, Line) and isinstance(b, Plane) or isinstance(b, Line) and isinstance(a, Plane):
            e = LeviCivitaTensor(n)
            result = TensorDiagram(*[(e, a)] * a.tensor_shape[1], *[(e, b)] * b.tensor_shape[1]).calculate()
        elif isinstance(a, Subspace) and isinstance(b, Point):
            result = a * b
        elif isinstance(a, Point) and isinstance(b, Subspace):
            result = b * a
        elif isinstance(a, Line) and isinstance(b, Line):
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
                        result = Tensor(array[i[0], ...], covariant=False, copy=False)
                    else:
                        # extract the point of intersection
                        result = Tensor(array[(slice(None),) + i[1:]], copy=False)
                else:
                    max_ind = np.abs(array).reshape((np.prod(array.shape[: coplanar.ndim]), -1)).argmax(1)
                    i = np.unravel_index(max_ind, array.shape[coplanar.ndim :])
                    i = tuple(np.reshape(x, array.shape[: coplanar.ndim]) for x in i)
                    indices = tuple(np.indices(array.shape[: coplanar.ndim]))
                    if not intersect_lines:
                        result_array = array[(*indices, i[0], Ellipsis)]
                        result_rank = result_array.ndim - coplanar.ndim
                        result = Tensor(result_array, covariant=False, tensor_rank=result_rank, copy=False)
                    else:
                        result = Tensor(array[indices + (slice(None),) + i[1:]], tensor_rank=1, copy=False)

            elif intersect_lines or n == 4:
                # can't intersect lines that are not coplanar and can't join skew lines in 3D
                raise NotCoplanar("The given lines are not all coplanar.")
            elif np.any(coplanar) and (a.cdim > 0 or b.cdim > 0):
                raise GeometryException("Can only join tensors that are either all coplanar or all not coplanar.")

        else:
            # TODO: intersect arbitrary subspaces (use GA)
            raise ValueError("Operation not supported.")

    else:
        raise ValueError("Wrong number of arguments.")

    if check_dependence:
        is_zero = result.is_zero()
        if result.cdim == 0 and is_zero:
            raise LinearDependenceError("Arguments are not linearly independent.")
        elif np.any(is_zero):
            raise LinearDependenceError("Some arguments are not linearly independent.", is_zero)

    if normalize_result:
        axes = tuple(result._covariant_indices) + tuple(result._contravariant_indices)
        max_abs = np.max(np.abs(result.array), axis=axes, keepdims=True)
        _, max_exponent = np.frexp(max_abs)
        result.array = _divide_by_power_of_two(result.array, max_exponent)

    if result.tensor_shape == (0, 1):
        return Line(result, copy=False) if n == 3 else Plane(result, copy=False)
    if result.tensor_shape == (1, 0):
        return Point(result, copy=False)
    if result.tensor_shape == (2, 0):
        return Line(result, copy=False).contravariant_tensor
    if result.tensor_shape == (0, n - 2):
        return Line(result, copy=False)

    return Subspace(result, copy=False)


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
def join(*args: Unpack[tuple[Point, Point]], _check_dependence: bool = True, _normalize_result: bool = True) -> Line:
    ...


@overload
def join(*args: Point | Subspace, _check_dependence: bool = True, _normalize_result: bool = True) -> Subspace:
    ...


def join(*args: Point | Subspace, _check_dependence: bool = True, _normalize_result: bool = True) -> Subspace:
    """Joins a number of objects to form a line, plane or subspace.

    Args:
        *args: Objects to join, e.g. 2 points, lines, a point and a line or 3 points.

    Returns:
        The resulting line, plane or subspace.

    Raises:
        LinearDependenceError: If two objects coincide.
        NotCoplanar: For two skew lines in 3D.

    """
    return _join_meet_duality(
        *args, intersect_lines=False, check_dependence=_check_dependence, normalize_result=_normalize_result
    )


@overload
def meet(*args: Line, _check_dependence: bool = True, _normalize_result: bool = True) -> Point:
    ...


@overload
def meet(*args: Unpack[tuple[Subspace, Line]], _check_dependence: bool = True, _normalize_result: bool = True) -> Point:
    ...


@overload
def meet(*args: Unpack[tuple[Line, Subspace]], _check_dependence: bool = True, _normalize_result: bool = True) -> Point:
    ...


def meet(*args: Subspace, _check_dependence: bool = True, _normalize_result: bool = True) -> Point | Subspace:
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


class Point(ProjectiveTensor):
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

    def __init__(self, *args: Tensor | npt.ArrayLike, homogenize=False, tensor_rank=1, **kwargs) -> None:
        if np.isscalar(args[0]):
            super().__init__(*args, 1, tensor_rank=tensor_rank, **kwargs)
            homogenize = False
        else:
            super().__init__(*args, tensor_rank=tensor_rank, **kwargs)
        if homogenize is True:
            self.array = np.append(self.array, np.ones(self.shape[:-1] + (1,), self.dtype), axis=-1)

    def __add__(self, other: Tensor | npt.ArrayLike) -> Tensor:
        if not isinstance(other, Point):
            return super().__add__(other)
        a, b = self.normalized_array, other.normalized_array
        result = a[..., :-1] + b[..., :-1]
        result = np.append(result, np.maximum(a[..., -1:], b[..., -1:]), axis=-1)
        return Point(result, copy=False)

    def __sub__(self, other: Tensor | npt.ArrayLike) -> Tensor:
        if not isinstance(other, Point):
            return super().__add__(other)
        a, b = self.normalized_array, other.normalized_array
        result = a[..., :-1] - b[..., :-1]
        result = np.append(result, np.maximum(a[..., -1:], b[..., -1:]), axis=-1)
        return Point(result, copy=False)

    def __mul__(self, other: Tensor | npt.ArrayLike) -> Tensor:
        if not np.isscalar(other):
            return super().__mul__(other)
        result = self.normalized_array[..., :-1] * other
        result = np.append(result, self.array[..., -1:] != 0, axis=-1)
        return Point(result, copy=False)

    def __truediv__(self, other: Tensor | npt.ArrayLike) -> Tensor:
        if not np.isscalar(other):
            return super().__truediv__(other)
        result = self.normalized_array[..., :-1] / other
        result = np.append(result, self.array[..., -1:] != 0, axis=-1)
        return Point(result, copy=False)

    def __repr__(self) -> str:
        if self.cdim == 0:
            result = f"Point({', '.join(self.normalized_array[:-1].astype(str))})"
            if self.isinf:
                result += " at Infinity"
            return result
        return f"PointCollection({self.normalized_array.tolist()})"

    @overload
    def join(self, *others: Unpack[tuple[Point, Point]]) -> Line:
        ...

    @overload
    def join(self, *others: Unpack[tuple[Point, Subspace]] | Unpack[tuple[Subspace, Point]]) -> Subspace:
        ...

    def join(self, *others: Point | Subspace) -> Subspace:
        return join(self, *others)

    def __getitem__(self, index: TensorIndex) -> Tensor | np.generic:
        result = super().__getitem__(index)

        if not isinstance(result, Tensor) or result.tensor_shape != (1, 0):
            return result

        return Point(result, copy=False)

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

    def _matrix_transform(self, m: npt.ArrayLike) -> Point:
        if self.cdim == 0:
            return Point(np.dot(m, self.array), copy=False)
        return Point(matvec(m, self.array), copy=False)

    @property
    def normalized_array(self) -> np.ndarray:
        """The coordinate array of the points with the last coordinates normalized to 1."""
        return self._normalize_array(self.array)

    @property
    def isinf(self) -> npt.NDArray[np.bool_]:
        """Boolean array that indicates which points lie at infinity."""
        return np.isclose(self.array[..., -1], 0, atol=EQ_TOL_ABS)

    @property
    def isreal(self) -> npt.NDArray[np.bool_]:
        """Boolean array that indicates which points are real."""
        return np.all(np.isreal(self.normalized_array), axis=-1)


class Subspace(ProjectiveTensor):
    """Represents a general subspace of a projective space. Line and Plane are subclasses.

    Args:
        *args: The coordinates of the subspace. Instead of separate coordinates, a single iterable can be supplied.
        tensor_rank: The rank of the tensors that represent the subspace(s). Default is 1.
        **kwargs: Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    """

    def __init__(self, *args: Tensor | npt.ArrayLike, tensor_rank: int = 1, **kwargs) -> None:
        kwargs.setdefault("covariant", False)
        super().__init__(*args, tensor_rank=tensor_rank, **kwargs)

    @overload
    def meet(self, other: Line) -> Point:
        ...

    @overload
    def meet(self, other: Subspace) -> Point | Subspace:
        ...

    def meet(self, other: Subspace) -> Point | Subspace:
        return meet(self, other)

    def join(self, *others: Point | Subspace) -> Subspace:
        return join(self, *others)

    def __getitem__(self, index: TensorIndex) -> Tensor | np.generic:
        result = super().__getitem__(index)

        if not isinstance(result, Tensor) or result.tensor_shape != self.tensor_shape:
            return result

        return type(self)(result, copy=False)

    def __add__(self, other: Tensor | npt.ArrayLike) -> Tensor:
        if not isinstance(other, Point):
            return super().__add__(other)

        from geometer.transformation import translation

        return translation(other).apply(self)

    def __sub__(self, other: Tensor | npt.ArrayLike) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other, copy=False)
        return self + (-other)

    @property
    def basis_matrix(self) -> np.ndarray:
        x = self.array
        x = x.reshape(x.shape[: self.cdim] + (-1, x.shape[-1]))
        return np.swapaxes(null_space(x, self.shape[-1] - self.rank + self.cdim), -1, -2)

    def _matrix_transform(self, m: npt.ArrayLike) -> Subspace:
        transformed_basis = matmul(self.basis_matrix, m, transpose_b=True)
        transformed_basis_points = [Point(p, copy=False) for p in np.swapaxes(transformed_basis, 0, -2)]
        return join(*transformed_basis_points)

    @property
    def general_point(self) -> Point:
        """Points in general position i.e. not in the subspaces."""
        n = self.dim + 1
        s = [self.shape[i] for i in self._collection_indices]
        p = Point(np.zeros([*s, n], dtype=int), copy=False)
        ind = np.ones(s, dtype=bool)
        for i in range(n):
            p[ind, -i - 1] = 1
            ind = self.contains(p)
            if not np.any(ind):
                break
        return p

    def parallel(self, through: Point) -> Subspace:
        """Returns the subspaces through given points that are parallel to this collection of subspaces.

        Args:
            through: The point through which the parallel subspaces are to be constructed.

        Returns:
            The parallel subspaces.

        """
        x = self.meet(infty_hyperplane(self.dim))
        return join(x, through)

    def contains(self, other: Point | Subspace, tol: float = EQ_TOL_ABS) -> npt.NDArray[np.bool_]:
        """Tests whether given points or lines lie in the subspaces.

        Args:
            other: The object(s) to test.
            tol: The accepted tolerance.

        Returns:
            Boolean array that indicates which of given points/lines lies in the subspaces.

        """
        if isinstance(other, Point):
            result = self * other

        elif isinstance(other, Line):
            result = self * other.covariant_tensor

        else:
            # TODO: test subspace
            raise ValueError(f"argument of type {type(other)} not supported")

        axes = tuple(result._covariant_indices) + tuple(result._contravariant_indices)
        return np.all(np.isclose(result.array, 0, atol=tol), axis=axes)

    def is_parallel(self, other: Subspace) -> npt.NDArray[np.bool_]:
        """Tests whether the given subspace is parallel to this subspace.

        Args:
            other: The other space to test.

        Returns:
            True, if the two subspaces are parallel.

        """
        x = self.meet(other)
        return infty_hyperplane(self.dim).contains(x)


class Line(Subspace):
    """Represents a line in a projective space of arbitrary dimension.

    Args:
        *args: Two points or the coordinates of the line. Instead of all coordinates separately, a single iterable can
            also be supplied.
        **kwargs: Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    """

    def __init__(self, *args: Tensor | npt.ArrayLike, **kwargs) -> None:
        if len(args) == 2:
            kwargs["copy"] = False
            p, q = args
            super().__init__(join(Point(p, copy=False), Point(q, copy=False)), tensor_rank=-2, **kwargs)
        else:
            super().__init__(*args, tensor_rank=-2, **kwargs)

    def _matrix_transform(self, m: npt.ArrayLike) -> Line:
        result = super()._matrix_transform(m)
        return cast(Line, result)

    def meet(self, other: Subspace) -> Point:
        result = super().meet(other)
        return cast(Point, result)

    @property
    def base_point(self) -> Point:
        """Base points for the lines, arbitrarily chosen."""
        if self.dim > 2:
            base = self.basis_matrix
            p, q = base[..., 0, :], base[..., 1, :]
            isinf = np.isclose(p[..., -1, None], 0, atol=EQ_TOL_ABS)
            result = np.where(isinf, q, p)
            return Point(result, copy=False)

        y_zero = np.isclose(self.array[..., 1], 0, atol=EQ_TOL_ABS)
        z_zero = np.isclose(self.array[..., 2], 0, atol=EQ_TOL_ABS)

        result = np.zeros_like(self.array)
        result[z_zero, 2] = 1

        result[~y_zero & ~z_zero, 1] = -self.array[~y_zero & ~z_zero, 2]
        result[~y_zero & ~z_zero, 2] = self.array[~y_zero & ~z_zero, 1]

        result[y_zero & ~z_zero, 0] = self.array[y_zero & ~z_zero, 2]
        result[y_zero & ~z_zero, 2] = -self.array[y_zero & ~z_zero, 0]

        return Point(result, copy=False)

    @property
    def direction(self) -> Point:
        """The direction of the lines (not normalized)."""
        if self.dim > 2:
            return meet(self, infty_hyperplane(self.dim), _normalize_result=False)

        x_zero = np.isclose(self.array[..., 0], 0, atol=EQ_TOL_ABS)
        y_zero = np.isclose(self.array[..., 1], 0, atol=EQ_TOL_ABS)

        result = np.zeros_like(self.array)
        result[x_zero & y_zero, 1] = 1

        result[~(x_zero & y_zero), 0] = self.array[~(x_zero & y_zero), 1]
        result[~(x_zero & y_zero), 1] = -self.array[~(x_zero & y_zero), 0]

        return Point(result, copy=False)

    @property
    def basis_matrix(self) -> np.ndarray:
        """A matrix with orthonormal basis vectors as rows."""
        if self.dim == 2:
            a = self.base_point.array
            b = np.cross(self.array, a)
            m = [a / np.linalg.norm(a, axis=-1, keepdims=True), b / np.linalg.norm(b, axis=-1, keepdims=True)]
            return np.stack(m, axis=-2)
        return super().basis_matrix

    @property
    def covariant_tensor(self) -> Line:
        """The covariant tensors of lines in 3D."""
        if self.dim != 3:
            raise NotImplementedError(f"Expected dimension 3 but found dimension {self.dim}.")
        if self.tensor_shape[0] > 0:
            return self
        e = LeviCivitaTensor(4)
        diagram = TensorDiagram((e, self), (e, self))
        return Line(diagram.calculate(), copy=False)

    @property
    def contravariant_tensor(self) -> Line:
        """The contravariant tensors of lines in 3D."""
        if self.dim != 3:
            raise NotImplementedError(f"Expected dimension 3 but found dimension {self.dim}.")
        if self.tensor_shape[1] > 0:
            return self
        e = LeviCivitaTensor(4, False)
        diagram = TensorDiagram((self, e), (self, e))
        return Line(diagram.calculate(), copy=False)

    def is_coplanar(self, other: Line) -> npt.NDArray[np.bool_]:
        """Tests whether another line lies in the same plane as this line, i.e. whether two lines intersect.

        Args:
            other: A line in 3D to test.

        Returns:
            True if the two lines intersect (i.e. they lie in the same plane).

        References:
          - Jim Blinn, Lines in Space: Back to the Diagrams, Line Intersections

        """
        if self.dim == 2:
            return np.ones(self.shape[: self.cdim], dtype=bool)

        e = LeviCivitaTensor(self.dim + 1)
        d = TensorDiagram(*[(e, self)] * (self.dim - 1), *[(e, other)] * (self.dim - 1))
        return d.calculate().is_zero()

    def mirror(self, pt: Point) -> Point:
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
                e = cast(Plane, e)
                f = e.perpendicular(self)
                return f.mirror(pt)

            m = e.basis_matrix
            arr = matvec(m, pt.array)
            arr_sort = np.argsort(np.abs(arr), axis=-1)
            arr_ind = tuple(np.indices(arr.shape)[:-1])
            m = m[(*arr_ind, arr_sort, slice(None))]
            pt = Point(arr[(*arr_ind, arr_sort)], copy=False)
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

    def perpendicular(self, through: Point, plane: Subspace | None = None) -> Line:
        """Construct the perpendicular line though a point.

        Args:
            through: The point through which the perpendicular is constructed.
            plane: In three or higher dimensional spaces, the 2-dimensional subspace that the perpendicular line is
                supposed to lie in, can be specified.

        Returns:
            The perpendicular line.

        """
        n = self.dim + 1
        contains = self.contains(through)
        result = Line(np.empty(contains.shape + (n,) * (n - 2), np.complex128))
        if np.any(contains):
            l = self
            if self.cdim > 0:
                l = self[contains]

            if n > 3:
                if plane is None:
                    # additional point is required to determine the exact line
                    plane = join(l, l.general_point)
                elif plane.cdim > 0:
                    plane = plane[contains]

                basis = plane.basis_matrix
                line_pts = matmul(l.basis_matrix, basis, transpose_b=True)
                l = Line(np.cross(line_pts[..., 0, :], line_pts[..., 1, :]), copy=False)

            p = Point(np.append(l.array[..., :-1], np.zeros(l.shape[:-1] + (1,), dtype=l.dtype), axis=-1), copy=False)

            if n > 3:
                p = p._matrix_transform(np.swapaxes(basis, -1, -2))

            result[contains] = join(through if through.cdim == 0 else through[contains], p)

        if np.any(~contains):
            if through.cdim > 0:
                through = through[~contains]
            if self.cdim > 0:
                result[~contains] = cast(Line, self[~contains]).mirror(through).join(through)
            else:
                result[~contains] = self.mirror(through).join(through)

        return Line(np.real_if_close(result.array), copy=False)

    def project(self, pt: Point) -> Point:
        """The orthogonal projection of points onto the lines.

        Args:
            pt: The points to project.

        Returns:
            The projected points.

        """
        l = self.perpendicular(pt)
        return self.meet(l)


class Plane(Subspace):
    """Represents a hyperplane in a projective space of arbitrary dimension.

    Args:
        *args: The points/lines spanning the plane or the coordinates of the hyperplane. Instead of separate
            coordinates, a single iterable can be supplied.
        **kwargs: Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    """

    def __init__(self, *args: Tensor | npt.ArrayLike, **kwargs) -> None:
        if all(isinstance(o, (Line, Point)) for o in args):
            kwargs["copy"] = False
            super().__init__(join(*args), **kwargs)
        else:
            super().__init__(*args, **kwargs)

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
        result[(*a, b)] = -self.array[(*ind_short, i.squeeze(), None)]
        q, r = np.linalg.qr(result)
        return np.swapaxes(q, -1, -2)

    def mirror(self, pt: Point) -> Point:
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
        l = Line(np.cross(basis[..., 0, :-1], basis[..., 1, :-1]), copy=False)
        p = l.base_point
        polars = Line(p.array, copy=False)

        from geometer.curve import absolute_conic

        p1, p2 = absolute_conic.intersect(polars)
        p1 = Point(np.append(p1.array, np.zeros(p1.shape[:-1] + (1,)), axis=-1), copy=False)
        p2 = Point(np.append(p2.array, np.zeros(p2.shape[:-1] + (1,)), axis=-1), copy=False)

        l1 = p1.join(pt)
        l2 = p2.join(pt)
        q1 = self.meet(l1)
        q2 = self.meet(l2)
        m1 = q1.join(p2)
        m2 = q2.join(p1)
        return m1.meet(m2)

    def project(self, pt: Point) -> Point:
        """The orthogonal projection of points onto the planes.

        Args:
            pt: The points to project.

        Returns:
            The projected points.

        """
        l = self.perpendicular(pt)
        return self.meet(l)

    @overload
    def perpendicular(self, through: Point) -> Line:
        ...

    @overload
    def perpendicular(self, through: Line) -> Plane:
        ...

    def perpendicular(self, through: Point | Line) -> Line | Plane:
        """Construct the perpendicular lines though the given points
        or the perpendicular planes through the given lines.

        Only works for lines in 3D.

        Args:
            through: The points or lines through which the perpendiculars are constructed.

        Returns:
            The perpendicular lines or planes.

        """
        if self.dim != 3 and isinstance(through, Line):
            raise NotImplementedError(f"Expected dimension 3 but found dimension {self.dim}.")

        p = self.array[..., :-1]
        p = Point(np.append(p, np.zeros(p.shape[:-1] + (1,), dtype=p.dtype), axis=-1), copy=False)
        return through.join(p)


I = Point([-1j, 1, 0])
J = Point([1j, 1, 0])
infty = Line(0, 0, 1)


def infty_hyperplane(dimension: int) -> Line | Plane:
    if dimension == 2:
        return infty
    return Plane([0] * dimension + [1])


infty_plane = infty_hyperplane(3)
