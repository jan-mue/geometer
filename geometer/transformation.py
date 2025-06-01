from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
import numpy.typing as npt
from typing_extensions import overload, override

from geometer.base import (
    BoundTensor,
    LeviCivitaTensor,
    ProjectiveTensor,
    Tensor,
    TensorCollection,
    TensorDiagram,
)
from geometer.exceptions import NoIncidence
from geometer.point import LineTensor, Point, Subspace, infty_hyperplane
from geometer.utils import inv, matmul, outer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Unpack

    from geometer.curve import Conic
    from geometer.utils.typing import NumericalScalarType, TensorParameters


@overload
def identity(dim: int, collection_dims: Literal[None] = ...) -> Transformation: ...


@overload
def identity(dim: int, collection_dims: tuple[int, ...] | None = ...) -> TransformationCollection: ...


def identity(dim: int, collection_dims: tuple[int, ...] | None = None) -> TransformationTensor:
    """Returns the identity transformation.

    Args:
        dim: The dimension of the projective space that the transformation acts on.
        collection_dims: Collection dimensions for a collection of identity transformations.
                         By default, only a single transformation is returned.

    Returns:
        The identity transformation(s).

    """
    if collection_dims is not None:
        e = np.eye(dim + 1)
        e = e.reshape((1,) * len(collection_dims) + e.shape)
        e = np.tile(e, (*collection_dims, 1, 1))
        return TransformationCollection(e, copy=None)
    return Transformation(np.eye(dim + 1), copy=None)


def affine_transform(matrix: npt.ArrayLike | None = None, offset: npt.ArrayLike = 0) -> Transformation:
    """Returns a projective transformation for the given affine transformation.

    Args:
        matrix: The transformation matrix.
        offset: The translation.

    Returns:
        The projective transformation that represents the affine transformation.

    """
    n = 2
    dtype: np.dtype[NumericalScalarType] = np.dtype(np.int_)

    if not np.isscalar(offset):
        offset = np.asarray(offset)
        n = offset.shape[0] + 1
        dtype = offset.dtype

    if matrix is not None:
        matrix = np.asarray(matrix)
        n = matrix.shape[0] + 1
        dtype = np.promote_types(dtype, matrix.dtype)

    result = np.eye(n, dtype=dtype)

    if matrix is not None:
        result[:-1, :-1] = matrix

    result[:-1, -1] = offset
    return Transformation(result, copy=None)


def rotation(angle: float | np.float64, axis: Point | None = None) -> Transformation:
    """Returns a projective transformation that represents a rotation by the specified angle (and axis).

    Args:
        angle: The angle to rotate by.
        axis: The axis to rotate around when rotating points in 3D.

    Returns:
        The rotation.

    References:
      - `Rotation matrix, Wikipedia <https://en.wikipedia.org/wiki/
        Rotation_matrix#Rotation_matrix_from_axis_and_angle>`_

    """
    if axis is None:
        return affine_transform([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    dimension = axis.dim
    e = LeviCivitaTensor(dimension, False)
    a = axis.normalized_array[:-1]
    a = a / np.linalg.norm(a)
    d = TensorDiagram(*[(Tensor(a, copy=None), e) for _ in range(dimension - 2)])
    u = d.calculate().array
    v = outer(a, a)
    result = np.cos(angle) * np.eye(dimension) + np.sin(angle) * u + (1 - np.cos(angle)) * v

    return affine_transform(result)


def translation(*coordinates: BoundTensor | npt.ArrayLike) -> Transformation:
    """Returns a projective transformation that represents a translation by the given coordinates.

    Args:
        *coordinates: The coordinates by which points are translated when applying the resulting transformation.

    Returns:
        The translation.

    """
    # TODO: support point collections
    offset = Point(*coordinates)
    return affine_transform(offset=offset.normalized_array[:-1])


def scaling(*factors: npt.ArrayLike) -> Transformation:
    """Returns a projective transformation that represents general scaling by given factors in each dimension.

    Args:
        *factors: The scaling factors by which each dimension is scaled.

    Returns:
        The scaling transformation.

    """
    return affine_transform(np.diag(factors))  # type: ignore[arg-type]


def reflection(axis: Subspace) -> Transformation:
    """Returns a projective transformation that represents a reflection at the given axis/hyperplane.

    Args:
        axis: The 2D-line or hyperplane to reflect points at.

    Returns:
        The reflection.

    References:
      - `Householder transformation, Wikipedia <https://en.wikipedia.org/wiki/Householder_transformation>`_

    """
    if axis == infty_hyperplane(axis.dim):
        return identity(axis.dim)

    v = axis.array[:-1]
    v = v / np.linalg.norm(v)  # type: ignore[operator]

    p = affine_transform(np.eye(axis.dim) - 2 * outer(v, v.conj()))

    base = axis.basis_matrix
    ind = base[:, -1].nonzero()[0][0]
    x = base[ind, :-1] / base[ind, -1]
    x = Point(*x)

    return translation(x) * p * translation(-x)  # type: ignore[return-value]


class TransformationTensor(ProjectiveTensor, ABC):
    """Represents a projective transformation in an arbitrary projective space.

    The underlying array is the matrix representation of the projective transformation. The matrix must be
    a non-singular square matrix of size n+1 when n is the dimension of the projective space.
    The transformation can be applied to a point or another object by multiplication.

    Args:
        *args: The array that defines the matrix representing the transformation.
        **kwargs: Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    """

    def __init__(self, *args: Tensor | npt.ArrayLike, **kwargs: Unpack[TensorParameters]) -> None:
        kwargs.setdefault("covariant", [0])
        super().__init__(*args, tensor_rank=2, **kwargs)
        if self.tensor_shape != (1, 1):
            raise ValueError(f"Expected tensor of type (1, 1), but is {self.tensor_shape}")
        if self.shape[-1] != self.shape[-2]:
            raise ValueError(f"Expected quadratic matrix, but last two dimensions are {self.shape[-2:]}")

    @override
    def __apply__(self, transformation: TransformationTensor) -> TransformationCollection | Transformation:
        return TransformationCollection.from_array(matmul(transformation.array, self.array))

    T = TypeVar("T", bound=Tensor)

    def apply(self, other: T) -> BoundTensor | TensorCollection | T:
        """Apply the transformation to another object.

        Args:
            other: The object to apply the transformation to.

        Returns:
            The result of applying this transformation to the supplied object.

        """
        if hasattr(other, "__apply__"):
            return other.__apply__(self)
        raise NotImplementedError(f"Object of type {type(other)} cannot be transformed.")

    @overload
    def __mul__(self, other: TransformationTensor) -> TransformationTensor: ...

    @overload
    def __mul__(self, other: T) -> T: ...

    @overload
    def __mul__(self, other: Tensor | npt.ArrayLike) -> Tensor: ...

    @override
    def __mul__(self, other: Tensor | npt.ArrayLike) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other, copy=False)
        try:
            return self.apply(other)
        except NotImplementedError:
            return super().__mul__(other)

    @override
    def __pow__(self, power: int, modulo: int | None = None) -> Tensor:
        if power == 0:
            if self.free_indices == 0:
                return identity(self.dim)
            return identity(self.dim, self.shape[: self.free_indices])
        if power < 0:
            return self.inverse().__pow__(-power, modulo)

        result = super().__pow__(power, modulo)
        return type(self)(result, copy=None)

    def inverse(self) -> TransformationTensor:
        """Calculates the inverse projective transformation.

        Returns:
            The inverse transformation.

        """
        return type(self)(inv(self.array), copy=None)


class Transformation(TransformationTensor, BoundTensor):
    @classmethod
    def from_points(cls, *args: tuple[Point, Point]) -> Transformation:
        """Constructs a projective transformation in n-dimensional projective space from the image of n + 2 points in general position.

        For two dimensional transformations, 4 pairs of points are required, of which no three points are collinear.
        For three dimensional transformations, 5 pairs of points are required, of which no four points are coplanar.

        Args:
            *args: Pairs of points, where in each pair one point is mapped to the other.

        Returns:
            The transformation mapping each of the given points to the specified points.

        References:
          - J. Richter-Gebert: Perspectives on Projective Geometry, Proof of Theorem 3.4

        """
        a = [x.array for x, y in args]
        b = [y.array for x, y in args]
        m1 = np.column_stack(a[:-1])
        m2 = np.column_stack(b[:-1])
        d1 = np.linalg.solve(m1, a[-1])  # type: ignore[arg-type]
        d2 = np.linalg.solve(m2, b[-1])  # type: ignore[arg-type]
        t1 = m1.dot(np.diag(d1))
        t2 = m2.dot(np.diag(d2))

        return cls(t2.dot(np.linalg.inv(t1)))

    @classmethod
    def from_points_and_conics(
        cls, points1: Sequence[Point], points2: Sequence[Point], conic1: Conic, conic2: Conic
    ) -> TransformationTensor:
        """Constructs a projective transformation from two conics and the image of pairs of 3 points on the conics.

        Args:
            points1: Source points on conic1.
            points2: Target points on conic2.
            conic1: Source quadric.
            conic2: Target quadric.

        Returns:
            The transformation that maps the points to each other and the conic to the other conic.

        References:
          - https://math.stackexchange.com/questions/654275/homography-between-ellipses

        """
        a1, b1, c1 = points1
        l1, l2 = conic1.tangent(a1), conic1.tangent(b1)

        if not isinstance(l1, LineTensor):
            raise NoIncidence(f"Point {a1} does not lie on the conic {conic1}.")
        if not isinstance(l2, LineTensor):
            raise NoIncidence(f"Point {b1} does not lie on the conic {conic1}.")

        m = l1.meet(l2).join(c1)
        p, q = conic1.intersect(m)
        d1 = p if q == c1 else q

        a2, b2, c2 = points2
        l1, l2 = conic2.tangent(a2), conic2.tangent(b2)

        if not isinstance(l1, LineTensor):
            raise NoIncidence(f"Point {a2} does not lie on the conic {conic2}.")
        if not isinstance(l2, LineTensor):
            raise NoIncidence(f"Point {b2} does not lie on the conic {conic2}.")

        m = l1.meet(l2).join(c2)
        p, q = conic2.intersect(m)
        d2 = p if q == c2 else q

        return cls.from_points((a1, a2), (b1, b2), (c1, c2), (d1, d2))

    T = TypeVar("T", bound=Tensor)

    @override
    def apply(self, other: T) -> T:
        return super().apply(other)  # type: ignore[return-value]


class TransformationCollection(TransformationTensor, TensorCollection[Transformation]):
    _element_class = Transformation
