from __future__ import annotations

import math
from abc import ABC
from itertools import combinations
from typing import TYPE_CHECKING, Union, cast

import numpy as np
import numpy.typing as npt
from numpy.lib.scimath import sqrt as csqrt

from geometer.base import (
    EQ_TOL_ABS,
    EQ_TOL_REL,
    NDArrayParameters,
    ProjectiveTensor,
    Tensor,
    TensorCollection,
    TensorDiagram,
)
from geometer.exceptions import IncidenceError, NotReducible
from geometer.point import I, J, LineTensor, PlaneTensor, PointTensor, SubspaceTensor, infty_plane, join
from geometer.transformation import rotation, translation
from geometer.utils import adjugate, det, hat_matrix, inv, is_multiple, matmul, matvec, outer, roots

if TYPE_CHECKING:
    from typing_extensions import Unpack


class QuadricTensor(ProjectiveTensor, ABC):
    r"""Represents a quadric, i.e. the zero set of a polynomial of degree 2, in any dimension.

    The quadric is defined by a symmetric matrix of size :math:`n+1` where :math:`n` is the dimension of the projective
    space. If :math:`A \in \mathbb{R}^{(n+1) \times (n+1)}`, the quadric contains all points
    :math:`x \in \mathbb{R}^{n+1}` such that :math:`x^T A x = 0`.

    Args:
        matrix: A two-dimensional array defining the (n+1)x(n+1) symmetric matrix of the quadric.
        is_dual: If true, the quadric represents a dual quadric, i.e. all hyperplanes tangent to the non-dual quadric.
        normalize_matrix: If true, normalize matrix using the (n+1)-th root of the absolute value of
            its pseudo-determinant.
        **kwargs: Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    Attributes:
        is_dual (bool): True if the quadric is a dual quadric
            i.e. contains all hyperplanes tangent to the non-dual quadric.

    """

    def __init__(
        self,
        matrix: Tensor | npt.ArrayLike,
        is_dual: bool = False,
        normalize_matrix: bool = False,
        **kwargs: Unpack[NDArrayParameters],
    ) -> None:
        self.is_dual = is_dual

        if normalize_matrix is True:
            matrix = matrix.array if isinstance(matrix, Tensor) else np.asarray(matrix)
            w = np.abs(np.linalg.eigvalsh(matrix))
            pseudo_det = np.prod(np.where(w > EQ_TOL_ABS, w, 1), axis=-1, keepdims=True)
            matrix = matrix / (pseudo_det ** (1 / matrix.shape[-1]))
            kwargs["copy"] = False

        if not is_dual:
            kwargs.setdefault("covariant", False)
        super().__init__(matrix, tensor_rank=2, **kwargs)

    def __add__(self, other: Tensor | npt.ArrayLike) -> Tensor:
        if not isinstance(other, PointTensor):
            return super().__add__(other)

        return translation(other).apply(self)

    def __sub__(self, other: Tensor | npt.ArrayLike) -> Tensor:
        if not isinstance(other, PointTensor):
            return super().__add__(other)

        return translation(-other).apply(self)

    @classmethod
    def from_planes(cls, e: PlaneTensor, f: PlaneTensor) -> QuadricTensor:
        """Construct a degenerate quadric from two hyperplanes.

        Args:
            e, f: The two planes the quadric consists of.

        Returns:
            The resulting quadric.

        """
        m = outer(e.array, f.array)
        m += m.T
        return QuadricTensor(m, normalize_matrix=True)

    def tangent(self, at: PointTensor) -> PlaneTensor:
        """Returns the hyperplane defining the tangent space at a given point.

        Args:
            at: A point on the quadric at which the tangent plane is calculated.

        Returns:
            The tangent plane at the given point.

        """
        return PlaneTensor(matvec(self.array, at.array), copy=False)

    def is_tangent(self, plane: SubspaceTensor) -> npt.NDArray[np.bool_]:
        """Tests if a given hyperplane is tangent to the quadric.

        Args:
            plane: The hyperplane to test.

        Returns:
            True if the given hyperplane is tangent to the quadric.

        """
        return self.dual.contains(plane)

    def contains(self, other: PointTensor | SubspaceTensor, tol: float = EQ_TOL_ABS) -> npt.NDArray[np.bool_]:
        """Tests if a given point lies on the quadric.

        Args:
            other: The point or hyperplane to test.
            tol: The accepted tolerance.

        Returns:
            True if the quadric contains the point.

        """
        if self.is_dual:
            d = TensorDiagram((self, other), (self, other.copy()))
        else:
            d = TensorDiagram((other, self), (other.copy(), self))
        return np.isclose(d.calculate().array, 0, atol=tol)

    @property
    def is_degenerate(self) -> npt.NDArray[np.bool_]:
        """True if the quadric is degenerate."""
        return np.isclose(det(self.array), 0, atol=EQ_TOL_ABS)

    @property
    def components(self) -> list[PointTensor | LineTensor | PlaneTensor]:
        """The components of a degenerate quadric."""
        # Algorithm adapted from Perspectives on Projective Geometry, Section 11.1
        n = self.shape[-1]
        indices = tuple(np.indices(self.shape[:-2]))

        if n == 3:
            b = adjugate(self.array)
            i = np.argmax(np.abs(np.diagonal(b, axis1=-2, axis2=-1)), axis=-1)
            beta = csqrt(-b[(*indices, i, i)])
            p = -b[(*indices, slice(None), i)] / np.where(beta != 0, beta, -1)[..., None]

        else:
            ind = np.indices((n, n))
            ind = np.stack(
                [np.delete(np.delete(ind, i, axis=1), i, axis=2) for i in combinations(range(n), n - 2)], axis=1
            )
            minors = det(self.array[..., ind[0], ind[1]])
            p = csqrt(-minors)

        # use the skew symmetric matrix m to get a matrix of rank 1 defining the same quadric
        m = hat_matrix(p)
        t = self.array + m

        # components are in the non-zero rows and columns (up to scalar multiple)
        i = np.unravel_index(np.abs(t).reshape(t.shape[:-2] + (-1,)).argmax(axis=-1), t.shape[-2:])
        p, q = t[indices + i[:1]], t[(*indices, slice(None), i[1])]

        if self.dim > 2 and not np.all(is_multiple(outer(q, p), t, rtol=EQ_TOL_REL, atol=EQ_TOL_ABS, axis=(-2, -1))):
            raise NotReducible("Quadric has no decomposition in 2 components.")

        # TODO: make order of components reproducible

        p, q = np.real_if_close(p), np.real_if_close(q)

        if self.is_dual:
            return [PointTensor(p, copy=False), PointTensor(q, copy=False)]
        elif n == 3:
            return [LineTensor(p, copy=False), LineTensor(q, copy=False)]
        return [PlaneTensor(p, copy=False), PlaneTensor(q, copy=False)]

    def intersect(self, other: LineTensor) -> list[PointTensor] | list[LineTensor]:
        """Calculates points of intersection of a line with the quadric.

        This method also returns complex points of intersection, even if the quadric and the line do not intersect in
        any real points.

        Args:
            other: The line to intersect this quadric with.

        Returns:
            The points of intersection.

        References:
          - J. Richter-Gebert: Perspectives on Projective Geometry, Section 11.3

        """
        reducible: bool | np.bool_ = np.all(self.is_degenerate)
        if reducible:
            try:
                e, f = self.components
            except NotReducible:
                reducible = False

        if not reducible:
            if self.dim > 2:
                arr = other.array.reshape(other.shape[: -other.tensor_shape[1]] + (-1, self.dim + 1))

                if other.free_indices == 0:
                    i = arr.nonzero()[0][0]
                    m = PlaneTensor(arr[i], copy=False).basis_matrix
                else:
                    i = np.any(arr, axis=-1).argmax(-1)
                    m = PlaneTensor(arr[(*tuple(np.indices(i.shape)), i)], copy=False).basis_matrix
                line = other._matrix_transform(m)
                projected_quadric = QuadricTensor(matmul(matmul(m, self.array), m, transpose_b=True), copy=False)
                return [
                    PointTensor(np.squeeze(matmul(np.expand_dims(point.array, -2), m), -2), copy=False)
                    for point in projected_quadric.intersect(line)
                ]
            else:
                m = hat_matrix(other.array)
                b = matmul(matmul(m, self.array, transpose_a=True), m)
                p, q = QuadricTensor(b, is_dual=not self.is_dual, copy=False).components
        else:
            if self.is_dual:
                e, f = cast(PointTensor, e), cast(PointTensor, f)
                p, q = e.join(other), f.join(other)
            else:
                e, f = cast(Union[LineTensor, PlaneTensor], e), cast(Union[LineTensor, PlaneTensor], f)
                p, q = e.meet(other), f.meet(other)

        if p.free_indices == 0 and p == q:
            return [p]

        return [p, q]

    @property
    def dual(self) -> QuadricTensor:
        """The dual quadric."""
        return QuadricTensor(inv(self.array), is_dual=not self.is_dual, copy=False)


class Quadric(QuadricTensor):
    pass


class QuadricCollection(QuadricTensor, TensorCollection[Quadric]):
    pass


class Conic(QuadricTensor):
    """A two-dimensional conic."""

    @classmethod
    def from_points(cls, a: PointTensor, b: PointTensor, c: PointTensor, d: PointTensor, e: PointTensor) -> Conic:
        """Construct a conic through five points.

        Args:
            a, b, c, d, e: The points lying on the conic.

        Returns:
            The resulting conic.

        """
        a, b, c, d, e = (
            a.normalized_array,
            b.normalized_array,
            c.normalized_array,
            d.normalized_array,
            e.normalized_array,
        )
        ace = det([a, c, e])
        bde = det([b, d, e])
        ade = det([a, d, e])
        bce = det([b, c, e])
        m = ace * bde * outer(np.cross(a, d), np.cross(b, c)) - ade * bce * outer(np.cross(a, c), np.cross(b, d))
        return Conic(np.real_if_close(m + m.T), normalize_matrix=True)

    @classmethod
    def from_lines(cls, g: LineTensor, h: LineTensor) -> Conic:
        """Construct a degenerate conic from two lines.

        Args:
            g, h: The two lines the conic consists of.

        Returns:
            The resulting conic.

        """
        m = outer(g.array, h.array)
        m += m.T
        return Conic(m, normalize_matrix=True)

    @classmethod
    def from_tangent(cls, tangent: LineTensor, a: PointTensor, b: PointTensor, c: PointTensor, d: PointTensor) -> Conic:
        """Construct a conic through four points and tangent to a line.

        Args:
            tangent: The line tangent to the conic.
            a, b, c, d: The points lying on the conic.

        Returns:
            The resulting conic.

        Raises:
            IncidenceError: If one of the points lies on the tangent.

        """
        if any(tangent.contains(p) for p in [a, b, c, d]):
            raise IncidenceError("The supplied points cannot lie on the supplied tangent!")

        a1, a2 = LineTensor(a, c).meet(tangent).normalized_array, LineTensor(b, d).meet(tangent).normalized_array
        b1, b2 = LineTensor(a, b).meet(tangent).normalized_array, LineTensor(c, d).meet(tangent).normalized_array

        o = tangent.general_point.array

        a2b1 = det([o, a2, b1])
        a2b2 = det([o, a2, b2])
        a1b1 = det([o, a1, b1])
        a1b2 = det([o, a1, b2])

        c1 = csqrt(a2b1 * a2b2)
        c2 = csqrt(a1b1 * a1b2)

        x = PointTensor(c1 * a1 + c2 * a2, copy=False)
        y = PointTensor(c1 * a1 - c2 * a2, copy=False)

        conic = cls.from_points(a, b, c, d, x)
        if np.all(np.isreal(conic.array)):
            return conic
        return cls.from_points(a, b, c, d, y)

    @classmethod
    def from_foci(cls, f1: PointTensor, f2: PointTensor, bound: PointTensor) -> Conic:
        """Construct a conic with the given focal points that passes through the boundary point.

        Args:
            f1, f2: The two focal points.
            bound: A boundary point that lies on the conic.

        Returns:
            The resulting conic.

        """
        t1 = join(f1, I, _normalize_result=False)
        t2 = join(f1, J, _normalize_result=False)
        t3 = join(f2, I, _normalize_result=False)
        t4 = join(f2, J, _normalize_result=False)
        p1, p2 = PointTensor(t1.array, copy=False), PointTensor(t2.array, copy=False)
        p3, p4 = PointTensor(t3.array, copy=False), PointTensor(t4.array, copy=False)
        c = cls.from_tangent(LineTensor(bound.array, copy=False), p1, p2, p3, p4)
        return Conic(np.linalg.inv(c.array), copy=False)

    @classmethod
    def from_crossratio(cls, cr: float, a: PointTensor, b: PointTensor, c: PointTensor, d: PointTensor) -> Conic:
        """Construct a conic from a cross ratio and four other points.

        This method relies on the fact that a point lies on a conic with five other points, if and only of the
        cross ratio seen from this point is the same as the cross ratio of four of the other points seen from the fifth
        point.

        Args:
            cr: The crossratio of the other points that defines the conic.
            a, b, c, d: The points lying on the conic.

        Returns:
            The resulting conic.

        References:
          - J. Richter-Gebert: Perspectives on Projective Geometry, Section 10.2

        """
        ac = adjugate([np.ones(3), a.array, c.array])[:, 0]
        bd = adjugate([np.ones(3), b.array, d.array])[:, 0]
        ad = adjugate([np.ones(3), a.array, d.array])[:, 0]
        bc = adjugate([np.ones(3), b.array, c.array])[:, 0]

        matrix = outer(ac, bd) - cr * outer(ad, bc)
        matrix += matrix.T

        return cls(matrix, normalize_matrix=True)

    def intersect(self, other: LineTensor | Conic) -> list[PointTensor]:
        """Calculates points of intersection with the conic.

        Args:
            other: The object to intersect this conic with.

        Returns:
            The points of intersection.

        References:
          - J. Richter-Gebert: Perspectives on Projective Geometry, Section 11.4

        """
        if isinstance(other, Conic):
            if other.is_degenerate:
                g, h = other.components
            else:
                a1, a2, a3 = self.array
                b1, b2, b3 = other.array
                alpha = det(self.array)
                beta = det([a1, a2, b3]) + det([a1, b2, a3]) + det([b1, a2, a3])
                gamma = det([a1, b2, b3]) + det([b1, a2, b3]) + det([b1, b2, a3])
                delta = det(other.array)

                sol = roots([alpha, beta, gamma, delta])

                c = Conic(sol[0] * self.array + other.array, is_dual=self.is_dual, copy=False)
                g, h = c.components

            result = self.intersect(g)
            result += [x for x in self.intersect(h) if x not in result]
            return result

        return super().intersect(other)

    def tangent(self, at: PointTensor) -> LineTensor | tuple[LineTensor, LineTensor]:
        """Calculates the tangent line at a given point or the tangent lines between a point and the conic.

        Args:
            at: The point to calculate the tangent at.

        Returns:
            The tangent line(s).

        """
        if self.contains(at):
            return self.polar(at)
        p, q = self.intersect(self.polar(at))
        return at.join(p), at.join(q)

    def polar(self, pt: PointTensor) -> LineTensor:
        """Calculates the polar line of the conic at a given point.

        Args:
            pt: The point to calculate the polar at.

        Returns:
            The polar line.

        """
        return LineTensor(self.array.dot(pt.array), copy=False)

    @property
    def foci(self) -> tuple[PointTensor, ...]:
        """The foci of the conic."""
        # Algorithm from Perspectives on Projective Geometry, Section 19.4
        i = self.tangent(at=I)
        j = self.tangent(at=J)

        if isinstance(i, LineTensor) and isinstance(j, LineTensor):
            return (i.meet(j),)

        i1, i2 = i
        j1, j2 = j
        f1, f2 = i1.meet(j1), i2.meet(j2)
        g1, g2 = i1.meet(j2), i2.meet(j1)

        if np.all(np.isreal(f1.normalized_array)):
            return f1, f2

        return g1, g2


absolute_conic = Conic(np.eye(3))


class Ellipse(Conic):
    """Represents an ellipse in 2D.

    Args:
        center: The center of the ellipse, default is Point(0, 0).
        hradius: The horizontal radius (along the x-axis), default is 1.
        vradius: The vertical radius (along the y-axis), default is 1.
        **kwargs: Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    """

    def __init__(
        self, center: PointTensor = PointTensor(0, 0), hradius: float = 1, vradius: float = 1, **kwargs
    ) -> None:
        if hradius == vradius == 0:
            raise ValueError("hradius and vradius can not both be zero.")

        r = np.array([vradius**2, hradius**2, 1])
        c = -center.normalized_array
        d = c * r
        m = np.eye(3, dtype=d.dtype)
        m[[0, 1], [0, 1]] = r[:2]
        m[2, :] = d
        m[:, 2] = d
        m[2, 2] = d.dot(c) - (r[0] * r[1] + 1)

        # normalize with abs(det(m))
        m = m / (np.prod(np.maximum(r[:2], 1))) ** (2 / 3)

        kwargs["copy"] = False
        super().__init__(m, **kwargs)


class Circle(Ellipse):
    """A circle in 2D.

    Args:
        center: The center point of the circle, default is Point(0, 0).
        radius: The radius of the circle, default is 1.
        **kwargs: Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    """

    def __init__(self, center: PointTensor = PointTensor(0, 0), radius: float = 1, **kwargs) -> None:
        if radius <= 0:
            raise ValueError(f"radius must be greater than 0, but is {radius}")
        super().__init__(center, radius, radius, **kwargs)

    @property
    def center(self) -> PointTensor:
        """The center of the circle."""
        return self.foci[0]

    @property
    def radius(self) -> float:
        """The radius of the circle."""
        c = self.array[:2, 2] / self.array[0, 0]
        return np.sqrt(c.dot(c) - self.array[2, 2] / self.array[0, 0])

    @property
    def lie_coordinates(self) -> np.ndarray:
        """The normalized Lie coordinates of the circle in R4."""
        m = self.center.normalized_array
        x = m[0] ** 2 + m[1] ** 2 - self.radius**2
        return np.array([(1 + x) / 2, (1 - x) / 2, m[0], m[1]]) / self.radius

    def intersection_angle(self, other: Circle) -> float:
        """Calculates the angle of intersection of two circles using its Lie coordinates.

        Args:
            other: The circle to intersect this circle with.

        Returns:
            The angle of intersection.

        References:
          - `Lie sphere geometry, Wikipedia <https://en.wikipedia.org/wiki/Lie_sphere_geometry>`_

        """
        # lorenz coordinates
        p1 = self.lie_coordinates
        p2 = other.lie_coordinates

        return np.arccos(np.vdot(p1, p2))

    @property
    def area(self) -> float:
        """The area of the circle."""
        return 2 * np.pi * self.radius**2


class Sphere(QuadricTensor):
    """A sphere in any dimension.

    Args:
        center: The center of the sphere, default is Point(0, 0, 0).
        radius: The radius of the sphere, default is 1.
        **kwargs: Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    """

    def __init__(
        self, center: PointTensor = PointTensor(0, 0, 0), radius: float = 1, **kwargs: Unpack[NDArrayParameters]
    ) -> None:
        if radius == 0:
            raise ValueError("Sphere radius cannot be 0.")

        c = -center.normalized_array
        m = np.eye(center.shape[0], dtype=np.promote_types(c.dtype, type(radius)))
        m[-1, :] = c
        m[:, -1] = c
        m[-1, -1] = c[:-1].dot(c[:-1]) - radius**2

        # normalize with abs(det(m))
        m = m / radius ** (2 / 3)

        kwargs["copy"] = False
        super().__init__(m, **kwargs)

    @property
    def center(self) -> PointTensor:
        """The center of the sphere."""
        return PointTensor(np.append(-self.array[:-1, -1], [self.array[0, 0]]), copy=False)

    @property
    def radius(self) -> float:
        """The radius of the sphere."""
        c = self.array[:-1, -1] / self.array[0, 0]
        return np.sqrt(c.dot(c) - self.array[-1, -1] / self.array[0, 0])

    @staticmethod
    def _alpha(n: int) -> float:
        return math.pi ** (n / 2) / math.gamma(n / 2 + 1)

    @property
    def volume(self) -> float:
        """The volume of the sphere."""
        n = self.dim
        return self._alpha(n) * self.radius**n

    @property
    def area(self) -> float:
        """The surface area of the sphere."""
        n = self.dim
        return n * self._alpha(n) * self.radius ** (n - 1)


class Cone(QuadricTensor):
    """A quadric that forms a circular double cone in 3D.

    Args:
        vertex: The vertex or apex of the cone. Default is (0, 0, 0).
        base_center: The center of the circle that forms the base of the cone. Default is (0, 0, 1)
        radius: The radius of the circle forming the base of the cone. Default is 1.
        **kwargs: Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    """

    def __init__(
        self,
        vertex: PointTensor = PointTensor(0, 0, 0),
        base_center: PointTensor = PointTensor(0, 0, 1),
        radius: float = 1,
        **kwargs: Unpack[NDArrayParameters],
    ) -> None:
        if radius == 0:
            raise ValueError("The radius of a cone can not be zero.")

        from geometer.operators import angle, dist

        h = dist(vertex, base_center)
        c = (radius / h) ** 2

        if np.isinf(h):
            # cone with vertex at infinity is a cylinder with the center of the base as center
            v = base_center.normalized_array
        else:
            v = vertex.normalized_array

        # first build a cone with axis parallel to the z-axis
        m = np.eye(4, dtype=np.promote_types(v.dtype, type(c)))
        m[-1, :] = -v
        m[:, -1] = -v

        if np.isinf(c):
            # if h == 0 the quadric becomes a circle
            m[3, 3] = v[:3].dot(v[:3]) - radius**2
        else:
            m[2:, 2:] *= -c
            m[3, 3] = v[:2].dot(v[:2]) - (radius**2 if np.isinf(h) else v[2] ** 2 * c)

        # rotate the axis of the cone
        v = PointTensor(v, copy=False)
        axis = LineTensor(v, v + PointTensor(0, 0, 1))
        new_axis = LineTensor(vertex, base_center)

        if new_axis != axis:
            a = angle(axis, new_axis)
            e = axis.join(new_axis)
            t = rotation(a, axis=PointTensor(*e.array[:3]))
            t = translation(v) * t * translation(-v)
            m = t.array.T.dot(m).dot(t.array)

        kwargs["normalize_matrix"] = True
        super().__init__(m, **kwargs)


class Cylinder(Cone):
    """An infinite circular cylinder in 3D.

    Args:
        center: The center of the cylinder. Default is (0, 0, 0).
        direction: The direction of the axis of the cylinder. Default is (0, 0, 1).
        radius: The radius of the cylinder. Default is 1.
        **kwargs: Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    """

    def __init__(
        self,
        center: PointTensor = PointTensor(0, 0, 0),
        direction: PointTensor = PointTensor(0, 0, 1),
        radius: float = 1,
        **kwargs: Unpack[NDArrayParameters],
    ) -> None:
        vertex = infty_plane.meet(LineTensor(center, center + direction))
        super().__init__(vertex, center, radius, **kwargs)
