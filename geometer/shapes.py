from __future__ import annotations

import math
from abc import ABC
from itertools import combinations
from typing import TYPE_CHECKING, TypeVar, cast

import numpy as np
import numpy.typing as npt
from typing_extensions import Self, override

from geometer.base import EQ_TOL_ABS, EQ_TOL_REL, Tensor, TensorCollection
from geometer.exceptions import IncompatibleShapeError, LinearDependenceError, NotCoplanar
from geometer.operators import angle, dist, harmonic_set
from geometer.point import (
    LineTensor,
    Plane,
    PlaneTensor,
    Point,
    PointCollection,
    PointLikeTensor,
    PointTensor,
    infty_hyperplane,
    join,
    meet,
)
from geometer.transformation import TransformationTensor, rotation, translation
from geometer.utils import det, distinct, is_multiple, matmul, matvec

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from geometer.utils.typing import NDArrayParameters, PolytopeParameters, TensorIndex


class PolytopeTensor(PointLikeTensor, ABC):
    """A class representing polytopes in arbitrary dimension.

    A (n+1)-polytope is a collection of n-polytopes that
    have some (n-1)-polytopes in common, where 3-polytopes are polyhedra, 2-polytopes are polygons, 1-polytopes are
    line segments and 0-polytopes are points.

    The polytope is stored as a multidimensional numpy array. Hence, all facets of the polytope must have the same
    number of vertices and facets.

    Args:
        *args: The polytopes defining the facets of the polytope.
        pdim: The dimension of the polytope. Default is 0, i.e. an instance of this class is a point.
        **kwargs: Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    Attributes:
        array (numpy.ndarray): The underlying numpy array.
        pdim (int): The dimension of the polytope.

    """

    pdim: int

    def __init__(self, *args: Tensor | npt.ArrayLike, pdim: int = 0, **kwargs: Unpack[NDArrayParameters]) -> None:
        self.pdim = pdim
        super().__init__(*args, **kwargs)

    @override
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(str(v) for v in self.vertices)})"

    @property
    def vertices(self) -> list[PointTensor]:
        """The vertices of the polytope."""
        first_polygon_index = self.rank - max(self.pdim - 1, 1) - 1
        new_shape = self.shape[:first_polygon_index] + (-1, self.shape[-1])
        array = self.array.reshape(new_shape)
        return list(distinct(PointCollection.from_array(x) for x in np.moveaxis(array, -2, 0)))

    @property
    def facets(self) -> list[PolytopeTensor] | list[PointTensor]:
        """The facets of the polytope."""
        first_polygon_index = self.rank - max(self.pdim - 1, 1) - 1
        slices = (slice(None),) * first_polygon_index
        return [self._cast_polytope(self[(*slices, i)], self.pdim - 1) for i in range(self.shape[first_polygon_index])]  # type: ignore[index, arg-type]

    @property
    def _edges(self) -> np.ndarray:
        v1 = self.array
        v2 = np.roll(v1, -1, axis=-2)
        return np.stack([v1, v2], axis=-2)

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PolytopeTensor):
            return super().__eq__(other)

        if self.shape != other.shape:
            return False

        if self.pdim > 2:
            # facets equal up to reordering
            facets1 = self.facets
            facets2 = other.facets
            return all(f in facets2 for f in facets1) and all(f in facets1 for f in facets2)

        # vertices equal up to circular reordering
        reversed_array = np.flip(other.array, axis=-2)
        for i in range(self.shape[-2]):
            if np.all(
                is_multiple(self.array, np.roll(other.array, i, axis=-2), axis=-1, rtol=EQ_TOL_REL, atol=EQ_TOL_ABS)
            ):
                return True
            if np.all(
                is_multiple(self.array, np.roll(reversed_array, i, axis=-2), axis=-1, rtol=EQ_TOL_REL, atol=EQ_TOL_ABS)
            ):
                return True

        return False

    @override
    def __add__(self, other: Tensor | npt.ArrayLike) -> Tensor:
        if not isinstance(other, PointTensor):
            return super().__add__(other)
        return translation(other).apply(self)

    @override
    def __sub__(self, other: Tensor | npt.ArrayLike) -> Tensor:
        if not isinstance(other, PointTensor):
            return super().__sub__(other)
        return self + (-other)

    @staticmethod
    def _cast_polytope(tensor: Tensor, pdim: int) -> PolytopeTensor:
        if pdim == 1:
            return SegmentCollection.from_tensor(tensor)
        if pdim == 2:
            if tensor.shape[-2] == 3:
                # TODO: check if a collection can be returned here
                return Triangle(tensor, copy=None)
            if tensor.shape[-2] == 4:
                try:
                    return Rectangle(tensor, copy=None)
                except NotCoplanar:
                    return PolytopeCollection.from_tensor(tensor)

            try:
                return PolygonCollection.from_tensor(tensor)
            except NotCoplanar:
                return PolytopeCollection.from_tensor(tensor)
        if pdim == 3:
            return Polyhedron(tensor, copy=None)

        return PolytopeCollection.from_tensor(tensor)

    @override
    def __getitem__(self, index: TensorIndex) -> Tensor | np.generic:
        result = super().__getitem__(index)

        if not isinstance(result, Tensor) or result.free_indices == 0 or result.tensor_shape != (1, 0):
            return result

        index_mapping = self._get_index_mapping(index)
        removed_cdims = [i for i in range(self.free_indices) if i not in index_mapping]
        first_polygon_index = self.rank - max(self.pdim - 1, 1) - 1
        result_pdim = self.pdim - len([i for i in removed_cdims if i >= first_polygon_index])

        if result_pdim == 0:
            return result

        return self._cast_polytope(result, result_pdim)


class Polytope(PolytopeTensor, ABC):
    pass


PolytopeT = TypeVar("PolytopeT", covariant=True, bound=Polytope)


class PolytopeCollection(PolytopeTensor, TensorCollection[PolytopeT], ABC):  # type: ignore[misc]
    @override
    def _validate_tensor(self) -> None:
        super()._validate_tensor()
        if self.free_indices <= max(self.pdim - 1, 1):
            raise IncompatibleShapeError(
                f"Tensor has not enough free indices and cannot be used in a {self.__class__.__name__}."
            )


class SegmentTensor(PolytopeTensor):
    """Represents a line segment in an arbitrary projective space.

    Segments with one point at infinity represent rays/half-lines in a traditional sense.

    Args:
        *args: The start and endpoint of the line segment, either as two Point objects or a single coordinate array.
        **kwargs: Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    """

    _line: LineTensor

    def __init__(self, *args: Tensor | npt.ArrayLike, **kwargs: Unpack[NDArrayParameters]) -> None:
        if len(args) == 2:
            a, b = args
            if isinstance(a, Tensor):
                a = a.array
            if isinstance(b, Tensor):
                b = b.array
            a, b = np.broadcast_arrays(a, b)
            kwargs["copy"] = False
            super().__init__(np.stack([a, b], axis=-2), pdim=1, **kwargs)
        else:
            super().__init__(*args, pdim=1, **kwargs)

        self._line = join(*self.vertices)

    @override
    def __apply__(self, transformation: TransformationTensor) -> Self | TensorCollection[Self]:
        result = super().__apply__(transformation)
        if not isinstance(result, type(self)):
            return result
        result._line = transformation.apply(result._line)  # type: ignore[assignment]
        return result

    @override
    def __getitem__(self, index: TensorIndex) -> Tensor | np.generic:
        result = super().__getitem__(index)

        if not isinstance(result, Tensor) or result.rank < 2 or result.shape[-2] != 2:
            return result

        return SegmentCollection.from_tensor(result)

    @override
    @property
    def vertices(self) -> list[PointTensor]:
        """The start and endpoint of the line segment."""
        a = PointCollection.from_array(self.array[..., 0, :])
        b = PointCollection.from_array(self.array[..., 1, :])
        return [a, b]

    @override
    @property
    def facets(self) -> list[PointTensor]:
        return self.vertices

    @override
    @property
    def _edges(self) -> np.ndarray:
        return self.array

    def contains(self, other: PointTensor, tol: float = EQ_TOL_ABS) -> npt.NDArray[np.bool_]:
        """Tests whether a point is contained in the segment.

        Args:
            other: The point to test.
            tol: The accepted tolerance.

        Returns:
            True if the point is contained in the segment.

        """
        if other.shape[0] == 0:
            return np.empty((0,), dtype=bool)

        result = self._line.contains(other)

        m = self.normalized_array
        arr = matmul(m, m, transpose_b=True)

        b = arr[..., 0]
        c = arr[..., 1]

        # TODO: only project points that lie on the lines
        d = other._matrix_transform(m).array

        # check that crossratio(c-b, b, c, d) is between 0 and 1
        b, c, d = np.broadcast_arrays(b, c, d)
        cd = det(np.stack([c, d], axis=-2))
        bd = det(np.stack([b, d], axis=-2))
        z = -bd
        w = cd - bd
        z_r, z_i = np.real(z), np.imag(z)
        w_r, w_i = np.real(w), np.imag(w)
        x = z_r * w_r + z_i * w_i
        y = w_r**2 + w_i**2
        x_zero = np.isclose(x, 0, atol=EQ_TOL_ABS)
        y_zero = np.isclose(y, 0, atol=EQ_TOL_ABS)
        return result & (~x_zero | ~y_zero) & (x + tol >= 0) & (x <= y + tol)

    def intersect(
        self, other: LineTensor | PlaneTensor | SegmentTensor | PolygonTensor | Polyhedron
    ) -> list[PointTensor]:
        """Intersect the line segment with another object.

        Args:
            other: The object to intersect the line segment with.

        Returns:
            The points of intersection.

        """
        if isinstance(other, (PolygonTensor, Polyhedron)):
            return other.intersect(self)

        if isinstance(other, SegmentTensor):
            result = meet(self._line, other._line, _check_dependence=False)
            ind = ~result.is_zero() & self.contains(result) & other.contains(result)
        else:
            result = meet(self._line, other, _check_dependence=False)
            ind = ~result.is_zero() & self.contains(result)

        if result.free_indices > 0:
            return list(cast(PointCollection, result[ind]))
        if ind:
            return [result]
        return []

    @property
    def midpoint(self) -> PointTensor:
        """The midpoint of the segment."""
        l = self._line.meet(infty_hyperplane(self.dim))
        return harmonic_set(*self.vertices, l)  # type: ignore[call-arg]

    @property
    def length(self) -> npt.NDArray[np.float64]:
        """The length of the segment."""
        return dist(*self.vertices)


class Segment(SegmentTensor, Polytope):
    pass


class SegmentCollection(SegmentTensor, PolytopeCollection[Segment]):  # type: ignore[misc]
    _element_class = Segment

    @override
    def expand_dims(self, axis: int) -> SegmentCollection:
        result = super().expand_dims(axis)
        result._line = result._line.expand_dims(axis - self.dim + 3 if axis < -1 else axis)  # type: ignore[attr-defined]
        return result


class Simplex(Polytope):
    """Represents a simplex in any dimension, i.e. a k-polytope with k+1 vertices where k is the dimension.

    The simplex determined by k+1 points is given by the convex hull of these points.

    Args:
        *args: The points that are the vertices of the simplex.
        **kwargs: Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    """

    # TODO: return only instances of type Simplex
    def __new__(cls, *args: Point, **kwargs: Unpack[PolytopeParameters]) -> PolytopeTensor:  # type: ignore[misc]
        expected_pdim = len(args) - 1
        pdim = kwargs.pop("pdim", expected_pdim)
        if pdim != expected_pdim:
            raise ValueError(f"A simplex with {len(args)} points must have pdim={expected_pdim}, but got pdim={pdim}.")

        if len(args) == 2:
            return Segment(*args, **kwargs)  # type: ignore[misc]

        return super(PolytopeTensor, cls).__new__(cls)

    def __init__(self, *args: Point, **kwargs: Unpack[PolytopeParameters]) -> None:
        kwargs.setdefault("pdim", len(args) - 1)
        if len(args) > 3:
            args = [Simplex(*x) for x in combinations(args, len(args) - 1)]  # type: ignore[assignment]
        super().__init__(*args, **kwargs)

    @property
    def volume(self) -> float:
        """The volume of the simplex, calculated using the Cayley-Menger determinant."""
        points = np.concatenate([v.array.reshape((1, v.shape[0])) for v in self.vertices], axis=0)
        points = self._normalize_array(points)
        n, k = points.shape

        if n == k:
            return 1 / math.factorial(n - 1) * abs(det(points))  # type: ignore[operator]

        indices = np.triu_indices(n)
        distances = points[indices[0]] - points[indices[1]]
        distances = np.sum(distances**2, axis=1)
        m = np.zeros((n + 1, n + 1), dtype=distances.dtype)
        m[indices] = distances
        m += m.T
        m[-1, :-1] = 1
        m[:-1, -1] = 1

        return np.sqrt((-1) ** n / (math.factorial(n - 1) ** 2 * 2 ** (n - 1)) * det(m))


class PolygonTensor(PolytopeTensor):
    """A flat polygon with vertices in any dimension.

    The vertices of the polygon must be given either in clockwise or counterclockwise order.

    Args:
        *args: The coplanar points that are the vertices of the polygon.
            They will be connected sequentially by line segments.
        **kwargs: Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    """

    _plane: PlaneTensor

    def __init__(self, *args: Tensor | npt.ArrayLike, **kwargs: Unpack[NDArrayParameters]) -> None:
        if all(isinstance(x, SegmentTensor) for x in args):
            args = tuple(s.array[..., 0, :] for s in args)  # type: ignore[union-attr]
        if len(args) > 1:
            args = tuple(a.array if isinstance(a, Tensor) else a for a in args)
            args = np.broadcast_arrays(*args)
            kwargs["copy"] = False
            args = (np.stack(args, axis=-2),)
        super().__init__(*args, pdim=2, **kwargs)
        self._plane = cast(PlaneTensor, join(*self.vertices[: self.dim]) if self.dim > 2 else None)

    @override
    @property
    def vertices(self) -> list[PointTensor]:
        return [PointCollection.from_array(self.array[..., i, :]) for i in range(self.shape[-2])]

    @override
    @property
    def facets(self) -> list[SegmentTensor]:
        return list(self.edges)

    @property
    def edges(self) -> SegmentCollection:
        """The edges of the polygon."""
        return SegmentCollection(self._edges, copy=None)

    def contains(self, other: PointTensor) -> npt.NDArray[np.bool_]:
        """Tests whether a point is contained in the polygon.

        Points on an edge of the polygon are considered True.

        Args:
            other: The point to test.

        Returns:
            True if the point is contained in the polygon.

        References:
          - http://paulbourke.net/geometry/polygonmesh/#insidepoly

        """
        if other.shape[0] == 0:
            return np.empty((0,), dtype=bool)

        if self.dim > 2:
            coplanar = self._plane.contains(other)

            if not np.any(coplanar):
                return coplanar

            # remove coordinates with the largest absolute value in the normal vector
            i = np.argmax(np.abs(self._plane.array[..., :-1]), axis=-1)
            i = np.where(self._plane.isinf, self.dim, i)
            i = np.expand_dims(i, -1)
            s = self.array.shape
            arr = np.delete(self.array, np.ravel_multi_index((*tuple(np.indices(s[:-1])), i), s))
            arr = arr.reshape(s[:-1] + (-1,))

            if isinstance(other, Point) and i.ndim == 1:
                other = Point(np.delete(other.array, i), copy=None)
            else:
                s = other.shape[:-1] + (1, other.shape[-1])
                other = np.delete(other.array, np.ravel_multi_index((*tuple(np.indices(s[:-1])), i), s))  # type: ignore[assignment]
                other = PointCollection(other.reshape(s[:-2] + (-1,)), copy=None)  # type: ignore[attr-defined]

            # TODO: only test coplanar points
            return coplanar & PolygonCollection.from_array(arr).contains(other)

        edges = self.edges
        edge_points = edges.contains(PointCollection(np.expand_dims(other.array, -2), copy=None))

        if isinstance(other, PointCollection):
            direction = np.zeros_like(other.array)
            ind = other.isinf
            direction[ind, 0] = other.array[ind, 1]
            direction[ind, 1] = -other.array[ind, 0]
            direction[~ind, 0] = 1
            rays = SegmentCollection(np.stack([other.array, direction], axis=-2), copy=None).expand_dims(-3)
        else:
            direction = [other.array[1], -other.array[0], 0] if other.isinf else [1, 0, 0]  # type: ignore[assignment]
            rays = Segment(np.stack([other.array, direction], axis=-2), copy=None)  # type: ignore[assignment]

        intersections = meet(edges._line, rays._line, _check_dependence=False)

        # ignore edges along the rays
        ray_edges = is_multiple(edges._line.array, rays._line.array, atol=EQ_TOL_ABS, rtol=EQ_TOL_REL, axis=-1)

        # ignore intersections of downward edges that end on the ray
        v1 = edges.array[..., 0, :]
        v2 = edges.array[..., 1, :]
        v1_intersections = (v1[..., 1] <= v2[..., 1]) & is_multiple(  # type: ignore[operator]
            intersections.array, v1, atol=EQ_TOL_ABS, rtol=EQ_TOL_REL, axis=-1
        )
        v2_intersections = (v2[..., 1] <= v1[..., 1]) & is_multiple(  # type: ignore[operator]
            intersections.array, v2, atol=EQ_TOL_ABS, rtol=EQ_TOL_REL, axis=-1
        )

        result = edges.contains(intersections)
        result &= rays.contains(intersections)
        result &= ~ray_edges & ~v1_intersections & ~v2_intersections
        result = np.sum(result, axis=-1) % 2 == 1
        result |= np.any(edge_points, axis=-1)

        return result

    def intersect(self, other: LineTensor | SegmentTensor) -> list[PointTensor]:
        """Intersect the polygon with another object.

        Args:
            other: The object to intersect the polygon with.

        Returns:
            The points of intersection.

        """
        if self.dim == 2:
            return list(distinct(self.edges.intersect(other)))

        if isinstance(other, SegmentTensor):
            try:
                result = self._plane.meet(other._line)
            except LinearDependenceError as e:
                if isinstance(other, SegmentTensor):
                    other = cast(SegmentTensor, other[~e.dependent_values])
                result = cast(PlaneTensor, self._plane[~e.dependent_values]).meet(other._line)
                return list(  # type: ignore[call-overload]
                    result[
                        PolygonCollection.from_tensor(self[~e.dependent_values]).contains(result)  # type: ignore[arg-type]
                        & other.contains(result)
                    ]
                )
            else:
                return list(result[self.contains(result) & other.contains(result)])  # type: ignore[call-overload]
        try:
            result = self._plane.meet(other)
        except LinearDependenceError as e:
            if other.free_indices > 0:
                other = cast(LineTensor | SegmentTensor, other[~e.dependent_values])
            result = cast(PlaneTensor, self._plane[~e.dependent_values]).meet(other)  # type: ignore[arg-type]
            return list(result[PolygonCollection.from_tensor(self[~e.dependent_values]).contains(result)])  # type: ignore[arg-type, call-overload]
        else:
            return list(result[self.contains(result)])  # type: ignore[call-overload]

    def _normalized_projection(self) -> np.ndarray:
        points = self.array

        if self.dim > 2:
            e = self._plane
            o = Point(*[0] * self.dim)
            if e.free_indices > 0:
                ind = ~e.contains(o)
                e[ind] = cast(PlaneTensor, e[ind]).parallel(o)
            elif not e.contains(o):
                # use parallel hyperplane for projection to avoid rescaling
                e = e.parallel(o)
            m = e.basis_matrix
            points = matvec(np.expand_dims(m, -3), points)

        return self._normalize_array(points)

    @property
    def area(self) -> npt.NDArray[np.float64]:
        """The area of the polygon."""
        points = self._normalized_projection()
        a = sum(det(points[..., [0, i, i + 1], :]) for i in range(1, points.shape[-2] - 1))
        return 1 / 2 * np.abs(a)

    @property
    def angles(self) -> list[npt.NDArray[np.float64]]:
        """The interior angles of the polygon."""
        result = []
        a = cast(Segment, self.edges[-1])
        for b in self.edges:
            result.append(angle(a.vertices[1], a.vertices[0], b.vertices[1]))
            a = b

        return result


class Polygon(PolygonTensor, Polytope):
    @property
    def centroid(self) -> Point:
        """The centroid (center of mass) of the polygon."""
        points = self.normalized_array
        centroids = [np.average(points[[0, i, i + 1], :-1], axis=0) for i in range(1, points.shape[0] - 1)]
        weights = [det(self._normalized_projection()[[0, i, i + 1]]) / 2 for i in range(1, points.shape[0] - 1)]
        return Point(*np.average(centroids, weights=weights, axis=0))


class PolygonCollection(PolygonTensor, PolytopeCollection[Polygon]):  # type: ignore[misc]
    _element_class = Polygon


class RegularPolygon(Polygon):
    """A class that can be used to construct regular polygon from a radius and a center point.

    Args:
        center: The center of the polygon.
        radius: The distance from the center to the vertices of the polygon.
        n: The number of vertices of the regular polygon.
        axis: If constructed in higher-dimensional spaces, an axis vector is required to orient the polygon.
        **kwargs: Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    """

    def __init__(
        self, center: Point, radius: float, n: int, axis: Point | None = None, **kwargs: Unpack[NDArrayParameters]
    ) -> None:
        if axis is None:
            p = Point(1, 0)
        else:
            e = Plane(np.append(axis.array[:-1], [0]), copy=None)
            p = Point(*e.basis_matrix[0, :-1], copy=None)

        vertex = center + radius * p

        vertices = []
        for i in range(n):
            t = rotation(2 * np.pi * i / n, axis=axis)
            t = translation(center) * t * translation(-center)  # type: ignore[assignment]
            vertices.append(t * vertex)

        super().__init__(*vertices, **kwargs)

    @property
    def radius(self) -> npt.NDArray[np.float64]:
        """The circumradius of the regular polygon."""
        return dist(self.center, self.vertices[0])

    @property
    def center(self) -> Point:
        """The center of the polygon."""
        return Point(*np.sum(self.normalized_array[:, :-1], axis=0))

    @property
    def inradius(self) -> npt.NDArray[np.float64]:
        """The inradius of the regular polygon."""
        return dist(self.center, cast(SegmentTensor, self.edges[0]).midpoint)


class Triangle(Polygon, Simplex):
    """A class representing triangles.

    Args:
        *args: The vertices of the triangle.
        **kwargs: Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    """

    def __init__(self, *args: Tensor | npt.ArrayLike, **kwargs: Unpack[NDArrayParameters]):
        super().__init__(*args, **kwargs)
        if self.shape[-2] != 3:
            raise ValueError(f"Triangle must have exactly 3 vertices, but got {self.shape[-2]} vertices.")

    @property
    def circumcenter(self) -> Point:
        """The circumcenter of the triangle."""
        e1, e2, e3 = self.edges
        bisector1 = e1._line.perpendicular(e1.midpoint, plane=self._plane)
        bisector2 = e2._line.perpendicular(e2.midpoint, plane=self._plane)
        return cast(Point, bisector1.meet(bisector2))

    @override
    def contains(self, other: PointTensor) -> npt.NDArray[np.bool_]:
        # faster algorithm using barycentric coordinates

        # TODO: vectorize

        a, b, c, p = np.broadcast_arrays(*self.array, other.array)

        lambda1 = det(np.stack([p, b, c], axis=-2))
        lambda2 = det(np.stack([a, p, c], axis=-2))

        result = (lambda1 <= 0) == (lambda2 <= 0)

        if not np.any(result):
            return result

        lambda3 = det(np.stack([a, b, p], axis=-2))

        area = lambda1 + lambda2 + lambda3

        if np.isscalar(area):
            if area < 0:  # type: ignore[operator]
                return lambda1 <= 0 and lambda3 <= 0  # type: ignore[return-value]
            return lambda1 >= 0 and lambda3 >= 0  # type: ignore[return-value]

        ind = area < 0
        result[ind] &= (lambda1[ind] <= 0) & (lambda3[ind] <= 0)  # type: ignore[index]
        result[~ind] &= (lambda1[~ind] >= 0) & (lambda3[~ind] >= 0)  # type: ignore[index]

        return result


class Rectangle(Polygon):
    """A class representing rectangles.

    Args:
        *args: The vertices of the rectangle.
        **kwargs: Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    """

    def __init__(self, *args: Tensor | npt.ArrayLike, **kwargs: Unpack[NDArrayParameters]) -> None:
        super().__init__(*args, **kwargs)
        if self.shape[-2] != 4:
            raise ValueError(f"Rectangle must have exactly 4 vertices, but got {self.shape[-2]} vertices.")


class Polyhedron(Polytope):
    """A class representing polyhedra (3-polytopes)."""

    def __init__(self, *args: Tensor | npt.ArrayLike, **kwargs: Unpack[NDArrayParameters]) -> None:
        super().__init__(*args, pdim=3, **kwargs)

    @property
    def faces(self) -> PolygonCollection:
        """The faces of the polyhedron."""
        return PolygonCollection(self.array, copy=None)

    @property
    def edges(self) -> list[SegmentTensor]:
        """The edges of the polyhedron."""
        result = self._edges
        return list(distinct(Segment(result[idx], copy=None) for idx in np.ndindex(*self.shape[:2])))

    @property
    def area(self) -> npt.NDArray[np.float64] | np.float64:
        """The surface area of the polyhedron."""
        return np.sum(self.faces.area)

    def intersect(self, other: LineTensor | SegmentTensor) -> list[PointTensor]:
        """Intersect the polyhedron with another object.

        Args:
            other: The object to intersect the polyhedron with.

        Returns:
            The points of intersection.

        """
        return list(distinct(self.faces.intersect(other)))


class Cuboid(Polyhedron):
    """A class that can be used to construct a cuboid/box or a cube.

    Args:
        a: The base point of the cuboid.
        b: The vertex that determines the first direction of the edges.
        c: The vertex that determines the second direction of the edges.
        d: The vertex that determines the third direction of the edges.
        **kwargs: Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    """

    def __init__(self, a: Point, b: Point, c: Point, d: Point, **kwargs: Unpack[NDArrayParameters]) -> None:
        x, y, z = b - a, c - a, d - a
        yz = Rectangle(a, a + z, a + y + z, a + y)
        xz = Rectangle(a, a + x, a + x + z, a + z)
        xy = Rectangle(a, a + x, a + x + y, a + y)
        super().__init__(yz, xz, xy, yz + x, xz + y, xy + z, **kwargs)
