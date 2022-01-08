import math
from itertools import combinations

import numpy as np
from numpy.lib.scimath import sqrt as csqrt

from .base import EQ_TOL_ABS, EQ_TOL_REL, ProjectiveCollection, ProjectiveElement, Tensor, TensorDiagram
from .exceptions import NotReducible
from .point import I, J, Line, LineCollection, Plane, PlaneCollection, Point, PointCollection, infty_plane, join
from .transformation import rotation, translation
from .utils import adjugate, det, hat_matrix, inv, is_multiple, matmul, roots


class Quadric(ProjectiveElement):
    r"""Represents a quadric, i.e. the zero set of a polynomial of degree 2, in any dimension.

    The quadric is defined by a symmetric matrix of size :math:`n+1` where :math:`n` is the dimension of the projective
    space. If :math:`A \in \mathbb{R}^{(n+1) \times (n+1)}`, the quadric contains all points
    :math:`x \in \mathbb{R}^{n+1}` such that :math:`x^T A x = 0`.

    Parameters
    ----------
    matrix : array_like or Tensor
        A two-dimensional array defining the (n+1)x(n+1) symmetric matrix of the quadric.
    is_dual : bool, optional
        If true, the quadric represents a dual quadric, i.e. all hyperplanes tangent to the non-dual quadric.
    normalize_matrix : bool, optional
        If true, normalize matrix using the (n+1)-th root of the absolute value of its pseudo-determinant.
    **kwargs
        Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    Attributes
    ----------
    is_dual : bool
        True if the quadric is a dual quadric i.e. contains all hyperplanes tangent to the non-dual quadric.

    """

    def __init__(self, matrix, is_dual=False, normalize_matrix=False, **kwargs):
        self.is_dual = is_dual

        if normalize_matrix is True:
            matrix = matrix.array if isinstance(matrix, Tensor) else np.asarray(matrix)
            w = np.abs(np.linalg.eigvalsh(matrix))
            matrix = matrix / np.prod(w[w > EQ_TOL_ABS]) ** (1 / matrix.shape[-1])
            kwargs["copy"] = False

        kwargs.setdefault("covariant", False)
        super(Quadric, self).__init__(matrix, **kwargs)

    def __add__(self, other):
        if not isinstance(other, Point):
            return super(Quadric, self).__add__(other)

        return translation(other).apply(self)

    def __sub__(self, other):
        return self + (-other)

    @classmethod
    def from_planes(cls, e, f):
        """Construct a degenerate quadric from two hyperplanes.

        Parameters
        ----------
        e, f : Plane
            The two planes the quadric consists of.

        Returns
        -------
        Quadric
            The resulting quadric.

        """
        m = np.outer(e.array, f.array)
        return Quadric(m + m.T, normalize_matrix=True)

    def tangent(self, at):
        """Returns the hyperplane defining the tangent space at a given point.

        Parameters
        ----------
        at : Point
            A point on the quadric at which the tangent plane is calculated.

        Returns
        -------
        Plane
            The tangent plane at the given point.

        """
        return Plane(self.array.dot(at.array), copy=False)

    def is_tangent(self, plane):
        """Tests if a given hyperplane is tangent to the quadric.

        Parameters
        ----------
        plane : Subspace
            The hyperplane to test.

        Returns
        -------
        bool
            True if the given hyperplane is tangent to the quadric.

        """
        return self.dual.contains(plane)

    def contains(self, other, tol=EQ_TOL_ABS):
        """Tests if a given point lies on the quadric.

        Parameters
        ----------
        other : Point or Subspace
            The point or hyperplane to test.
        tol : float, optional
            The accepted tolerance.

        Returns
        -------
        bool
            True if the quadric contains the point.

        """
        return np.isclose(other.array.dot(self.array.dot(other.array)), 0, atol=tol)

    @property
    def is_degenerate(self):
        """bool: True if the quadric is degenerate."""
        return np.isclose(det(self.array), 0, atol=EQ_TOL_ABS)

    @property
    def components(self):
        """list of ProjectiveElement: The components of a degenerate quadric."""
        # Algorithm adapted from Perspectives on Projective Geometry, Section 11.1
        n = self.shape[0]

        if n == 3:
            b = adjugate(self.array)
            i = np.argmax(np.abs(np.diag(b)))
            beta = csqrt(-b[i, i])
            p = -b[:, i] / beta if beta != 0 else b[:, i]

        else:
            p = []
            for ind in combinations(range(n), n - 2):
                # calculate all principal minors of order 2
                row_ind = [[j] for j in range(n) if j not in ind]
                col_ind = [j for j in range(n) if j not in ind]
                p.append(csqrt(-det(self.array[row_ind, col_ind])))

        # use the skew symmetric matrix m to get a matrix of rank 1 defining the same quadric
        m = hat_matrix(p)
        t = self.array + m

        # components are in the non-zero rows and columns (up to scalar multiple)
        i = np.unravel_index(np.abs(t).argmax(), t.shape)
        p, q = t[i[0]], t[:, i[1]]

        if self.dim > 2 and not is_multiple(np.outer(q, p), t, rtol=EQ_TOL_REL, atol=EQ_TOL_ABS):
            raise NotReducible("Quadric has no decomposition in 2 components.")

        p, q = np.real_if_close(p), np.real_if_close(q)

        if self.is_dual:
            return [Point(p, copy=False), Point(q, copy=False)]
        elif n == 3:
            return [Line(p, copy=False), Line(q, copy=False)]
        return [Plane(p, copy=False), Plane(q, copy=False)]

    def intersect(self, other):
        """Calculates points of intersection of a line with the quadric.

        This method also returns complex points of intersection, even if the quadric and the line do not intersect in
        any real points.

        Parameters
        ----------
        other: Line or LineCollection
            The line to intersect this quadric with.

        Returns
        -------
        list of Point or list of PointCollection
            The points of intersection.

        References
        ----------
        .. [1] J. Richter-Gebert: Perspectives on Projective Geometry, Section 11.3

        """
        if isinstance(other, (Line, LineCollection)):
            reducible = self.is_degenerate
            if reducible:
                try:
                    e, f = self.components
                except NotReducible:
                    reducible = False

            if not reducible:
                if self.dim > 2:
                    arr = other.array.reshape(other.shape[: -other.tensor_shape[1]] + (-1, self.dim + 1))

                    if isinstance(other, Line):
                        i = arr.nonzero()[0][0]
                        m = Plane(arr[i], copy=False).basis_matrix
                        q = Quadric(m.dot(self.array).dot(m.T), copy=False)
                        line = other._matrix_transform(m)
                        return [Point(m.T.dot(p.array), copy=False) for p in q.intersect(line)]
                    else:
                        i = np.any(arr, axis=-1).argmax(-1)
                        m = PlaneCollection(arr[tuple(np.indices(i.shape)) + (i,)], copy=False).basis_matrix
                        line = other._matrix_transform(m)
                        q = QuadricCollection(matmul(m.dot(self.array), m, transpose_b=True), copy=False)
                        return [
                            PointCollection(np.squeeze(matmul(np.expand_dims(p.array, -2), m), -2), copy=False)
                            for p in q.intersect(line)
                        ]
                else:
                    m = hat_matrix(other.array)
                    b = matmul(np.swapaxes(m, -1, -2).dot(self.array), m)

                    if isinstance(other, Line):
                        p, q = Conic(b, is_dual=not self.is_dual, copy=False).components
                    else:
                        p, q = QuadricCollection(b, is_dual=not self.is_dual, copy=False).components
            else:
                if self.is_dual:
                    p, q = e.join(other), f.join(other)
                else:
                    p, q = e.meet(other), f.meet(other)

            if p == q:
                return [p]

            return [p, q]

    @property
    def dual(self):
        """Quadric: The dual quadric."""
        return Quadric(np.linalg.inv(self.array), is_dual=not self.is_dual, copy=False)


class Conic(Quadric):
    """A two-dimensional conic."""

    @classmethod
    def from_points(cls, a, b, c, d, e):
        """Construct a conic through five points.

        Parameters
        ----------
        a, b, c, d, e : Point
            The points lying on the conic.

        Returns
        -------
        Conic
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
        m = ace * bde * np.outer(np.cross(a, d), np.cross(b, c)) - ade * bce * np.outer(np.cross(a, c), np.cross(b, d))
        return Conic(np.real_if_close(m + m.T), normalize_matrix=True)

    @classmethod
    def from_lines(cls, g, h):
        """Construct a degenerate conic from two lines.

        Parameters
        ----------
        g, h : Line
            The two lines the conic consists of.

        Returns
        -------
        Conic
            The resulting conic.

        """
        m = np.outer(g.array, h.array)
        return Conic(m + m.T, normalize_matrix=True)

    @classmethod
    def from_tangent(cls, tangent, a, b, c, d):
        """Construct a conic through four points and tangent to a line.

        Parameters
        ----------
        tangent : Line
        a, b, c, d : Point
            The points lying on the conic.

        Returns
        -------
        Conic
            The resulting conic.

        """
        if any(tangent.contains(p) for p in [a, b, c, d]):
            raise ValueError("The supplied points cannot lie on the supplied tangent!")

        a1, a2 = Line(a, c).meet(tangent).normalized_array, Line(b, d).meet(tangent).normalized_array
        b1, b2 = Line(a, b).meet(tangent).normalized_array, Line(c, d).meet(tangent).normalized_array

        o = tangent.general_point.array

        a2b1 = det([o, a2, b1])
        a2b2 = det([o, a2, b2])
        a1b1 = det([o, a1, b1])
        a1b2 = det([o, a1, b2])

        c1 = csqrt(a2b1 * a2b2)
        c2 = csqrt(a1b1 * a1b2)

        x = Point(c1 * a1 + c2 * a2, copy=False)
        y = Point(c1 * a1 - c2 * a2, copy=False)

        conic = cls.from_points(a, b, c, d, x)
        if np.all(np.isreal(conic.array)):
            return conic
        return cls.from_points(a, b, c, d, y)

    @classmethod
    def from_foci(cls, f1, f2, bound):
        """Construct a conic with the given focal points that passes through the boundary point.

        Parameters
        ----------
        f1, f2 : Point
            The two focal points.
        bound : Point
            A boundary point that lies on the conic.

        Returns
        -------
        Conic
            The resulting conic.

        """
        t1 = join(f1, I, _normalize_result=False)
        t2 = join(f1, J, _normalize_result=False)
        t3 = join(f2, I, _normalize_result=False)
        t4 = join(f2, J, _normalize_result=False)
        p1, p2 = Point(t1.array, copy=False), Point(t2.array, copy=False)
        p3, p4 = Point(t3.array, copy=False), Point(t4.array, copy=False)
        c = cls.from_tangent(Line(bound.array, copy=False), p1, p2, p3, p4)
        return Conic(np.linalg.inv(c.array), copy=False)

    @classmethod
    def from_crossratio(cls, cr, a, b, c, d):
        """Construct a conic from a cross ratio and four other points.

        This method relies on the fact that a point lies on a conic with five other points, if and only of the
        cross ratio seen from this point is the same as the cross ratio of four of the other points seen from the fifth
        point.

        Parameters
        ----------
        cr : float
            The crossratio of the other points that defines the conic.
        a, b, c, d : Point
            The points lying on the conic.

        Returns
        -------
        Conic
            The resulting conic.

        References
        ----------
        .. [1] J. Richter-Gebert: Perspectives on Projective Geometry, Section 10.2

        """
        ac = adjugate([np.ones(3), a.array, c.array])[:, 0]
        bd = adjugate([np.ones(3), b.array, d.array])[:, 0]
        ad = adjugate([np.ones(3), a.array, d.array])[:, 0]
        bc = adjugate([np.ones(3), b.array, c.array])[:, 0]

        matrix = np.outer(ac, bd) - cr * np.outer(ad, bc)

        return cls(matrix + matrix.T, normalize_matrix=True)

    def intersect(self, other):
        """Calculates points of intersection with the conic.

        Parameters
        ----------
        other: Line, LineCollection or Conic
            The object to intersect this conic with.

        Returns
        -------
        list of Point or list of PointCollection
            The points of intersection.

        References
        ----------
        .. [1] J. Richter-Gebert: Perspectives on Projective Geometry, Section 11.4

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

        return super(Conic, self).intersect(other)

    def tangent(self, at):
        """Calculates the tangent line at a given point or the tangent lines between a point and the conic.

        Parameters
        ----------
        at : Point
            The point to calculate the tangent at.

        Returns
        -------
        Line or tuple of Line
            The tangent line(s).

        """
        if self.contains(at):
            return self.polar(at)
        p, q = self.intersect(self.polar(at))
        return at.join(p), at.join(q)

    def polar(self, pt):
        """Calculates the polar line of the conic at a given point.

        Parameters
        ----------
        pt : Point
            The point to calculate the polar at.

        Returns
        -------
        Line
            The polar line.

        """
        return Line(self.array.dot(pt.array), copy=False)

    @property
    def foci(self):
        """tuple of Point: The foci of the conic."""
        # Algorithm from Perspectives on Projective Geometry, Section 19.4
        i = self.tangent(at=I)
        j = self.tangent(at=J)

        if isinstance(i, Line) and isinstance(j, Line):
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

    Parameters
    ----------
    center : Point, optional
        The center of the ellipse, default is Point(0, 0).
    hradius : float, optional
        The horizontal radius (along the x-axis), default is 1.
    vradius : float, optional
         The vertical radius (along the y-axis), default is 1.
    **kwargs
        Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    """

    def __init__(self, center=Point(0, 0), hradius=1, vradius=1, **kwargs):
        if hradius == vradius == 0:
            raise ValueError("hradius and vradius can not both be zero.")

        r = np.array([vradius ** 2, hradius ** 2, 1])
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
        super(Ellipse, self).__init__(m, **kwargs)


class Circle(Ellipse):
    """A circle in 2D.

    Parameters
    ----------
    center : Point, optional
        The center point of the circle, default is Point(0, 0).
    radius : float, optional
        The radius of the circle, default is 1.
    **kwargs
        Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    """

    def __init__(self, center=Point(0, 0), radius=1, **kwargs):
        super(Circle, self).__init__(center, radius, radius, **kwargs)

    @property
    def center(self):
        """Point: The center of the circle."""
        return self.foci[0]

    @property
    def radius(self):
        """float: The radius of the circle."""
        c = self.array[:2, 2] / self.array[0, 0]
        return np.sqrt(c.dot(c) - self.array[2, 2] / self.array[0, 0])

    @property
    def lie_coordinates(self):
        """Point: The Lie coordinates of the circle as point in RP4."""
        m = self.center.normalized_array
        x = m[0] ** 2 + m[1] ** 2 - self.radius ** 2
        return Point([(1 + x) / 2, (1 - x) / 2, m[0], m[1], self.radius])

    def intersection_angle(self, other):
        """Calculates the angle of intersection of two circles using its Lie coordinates.

        Parameters
        ----------
        other : Circle
            The circle to intersect this circle with.

        Returns
        -------
        float
            The angle of intersection.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Lie_sphere_geometry

        """
        # lorenz coordinates
        p1 = self.lie_coordinates.normalized_array[:-1]
        p2 = other.lie_coordinates.normalized_array[:-1]

        return np.arccos(np.vdot(p1, p2))

    @property
    def area(self):
        """float: The area of the circle."""
        return 2 * np.pi * self.radius ** 2


class Sphere(Quadric):
    """A sphere in any dimension.

    Parameters
    ----------
    center : Point, optional
        The center of the sphere, default is Point(0, 0, 0).
    radius : float, optional
        The radius of the sphere, default is 1.
    **kwargs
        Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    """

    def __init__(self, center=Point(0, 0, 0), radius=1, **kwargs):
        if radius == 0:
            raise ValueError("Sphere radius cannot be 0.")

        c = -center.normalized_array
        m = np.eye(center.shape[0], dtype=np.find_common_type([c.dtype, type(radius)], []))
        m[-1, :] = c
        m[:, -1] = c
        m[-1, -1] = c[:-1].dot(c[:-1]) - radius ** 2

        # normalize with abs(det(m))
        m = m / radius ** (2 / 3)

        kwargs["copy"] = False
        super(Sphere, self).__init__(m, **kwargs)

    @property
    def center(self):
        """Point: The center of the sphere."""
        return Point(np.append(-self.array[:-1, -1], [self.array[0, 0]]), copy=False)

    @property
    def radius(self):
        """float: The radius of the sphere."""
        c = self.array[:-1, -1] / self.array[0, 0]
        return np.sqrt(c.dot(c) - self.array[-1, -1] / self.array[0, 0])

    @staticmethod
    def _alpha(n):
        return math.pi ** (n / 2) / math.gamma(n / 2 + 1)

    @property
    def volume(self):
        """float: The volume of the sphere."""
        n = self.dim
        return self._alpha(n) * self.radius ** n

    @property
    def area(self):
        """float: The surface area of the sphere."""
        n = self.dim
        return n * self._alpha(n) * self.radius ** (n - 1)


class Cone(Quadric):
    """A quadric that forms a circular double cone in 3D.

    Parameters
    ----------
    vertex : Point, optional
        The vertex or apex of the cone. Default is (0, 0, 0).
    base_center : Point, optional
        The center of the circle that forms the base of the cone. Default is (0, 0, 1)
    radius : float, optional
        The radius of the circle forming the base of the cone. Default is 1.
    **kwargs
        Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    """

    def __init__(self, vertex=Point(0, 0, 0), base_center=Point(0, 0, 1), radius=1, **kwargs):
        if radius == 0:
            raise ValueError("The radius of a cone can not be zero.")

        from .operators import angle, dist

        h = dist(vertex, base_center)
        c = (radius / h) ** 2

        if np.isinf(h):
            # cone with vertex at infinity is a cylinder with the center of the base as center
            v = base_center.normalized_array
        else:
            v = vertex.normalized_array

        # first build a cone with axis parallel to the z-axis
        m = np.eye(4, dtype=np.find_common_type([v.dtype, type(c)], []))
        m[-1, :] = -v
        m[:, -1] = -v

        if np.isinf(c):
            # if h == 0 the quadric becomes a circle
            m[3, 3] = v[:3].dot(v[:3]) - radius ** 2
        else:
            m[2:, 2:] *= -c
            m[3, 3] = v[:2].dot(v[:2]) - (radius ** 2 if np.isinf(h) else v[2] ** 2 * c)

        # rotate the axis of the cone
        v = Point(v, copy=False)
        axis = Line(v, v + Point(0, 0, 1))
        new_axis = Line(vertex, base_center)

        if new_axis != axis:
            a = angle(axis, new_axis)
            e = axis.join(new_axis)
            t = rotation(a, axis=Point(*e.array[:3]))
            t = translation(v) * t * translation(-v)
            m = t.array.T.dot(m).dot(t.array)

        kwargs["normalize_matrix"] = True
        super(Cone, self).__init__(m, **kwargs)


class Cylinder(Cone):
    """An infinite circular cylinder in 3D.

    Parameters
    ----------
    center : Point, optional
        The center of the cylinder. Default is (0, 0, 0).
    direction : Point
        The direction of the axis of the cylinder. Default is (0, 0, 1).
    radius : float, optional
        The radius of the cylinder. Default is 1.
    **kwargs
        Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    """

    def __init__(self, center=Point(0, 0, 0), direction=Point(0, 0, 1), radius=1, **kwargs):
        vertex = infty_plane.meet(Line(center, center + direction))
        super(Cylinder, self).__init__(vertex, center, radius, **kwargs)


class QuadricCollection(ProjectiveCollection):
    """A collection of quadrics or conics."""

    def __init__(self, matrices, is_dual=False, **kwargs):
        self.is_dual = is_dual

        if not is_dual:
            kwargs.setdefault("covariant", False)
        super(QuadricCollection, self).__init__(matrices, tensor_rank=2, **kwargs)

    def tangent(self, at):
        """Returns the hyperplanes defining the tangent spaces at given points.

        Parameters
        ----------
        at : Point or PointCollection
            A point on the quadric at which the tangent plane is calculated.

        Returns
        -------
        PlaneCollection
            The tangent planes at the given points.

        """
        return PlaneCollection(self * at, copy=False)

    def is_tangent(self, planes):
        """Tests if a given hyperplane is tangent to the quadrics.

        Parameters
        ----------
        planes : Subspace or SubspaceCollection
            The hyperplane to test.

        Returns
        -------
        numpy.ndarray
            Returns a boolean array of which hyperplanes are tangent to the quadrics.

        """
        return self.dual.contains(planes)

    def contains(self, other, tol=EQ_TOL_ABS):
        """Tests if a given point lies on the quadrics.

        Parameters
        ----------
        other : Point, PointCollection, Subspace or SubspaceCollection
            The points to test.
        tol : float, optional
            The accepted tolerance.

        Returns
        -------
        numpy.ndarray
            Returns a boolean array of which quadrics contain the points.

        """
        if self.is_dual:
            d = TensorDiagram((self, other), (self, other.copy()))
        else:
            d = TensorDiagram((other, self), (other.copy(), self))
        return np.isclose(d.calculate().array, 0, atol=tol)

    @property
    def is_degenerate(self):
        """numpy.ndarray: Boolean array of which quadrics are degenerate in the collection."""
        return np.isclose(det(self.array), 0, atol=EQ_TOL_ABS)

    @property
    def components(self):
        """list of ProjectiveCollection: The components of the degenerate quadrics."""
        n = self.shape[-1]
        indices = tuple(np.indices(self.shape[:-2]))

        if n == 3:
            b = adjugate(self.array)
            i = np.argmax(np.abs(np.diagonal(b, axis1=-2, axis2=-1)), axis=-1)
            beta = csqrt(-b[indices + (i, i)])
            p = (-b[indices + (slice(None), i)] / np.where(beta != 0, beta, -1)[..., None])

        else:
            ind = np.indices((n, n))
            ind = [
                np.delete(np.delete(ind, i, axis=1), i, axis=2)
                for i in combinations(range(n), n - 2)
            ]
            ind = np.stack(ind, axis=1)
            minors = det(self.array[..., ind[0], ind[1]])
            p = csqrt(-minors)

        # use the skew symmetric matrix m to get a matrix of rank 1 defining the same quadric
        m = hat_matrix(p)
        t = self.array + m

        # components are in the non-zero rows and columns (up to scalar multiple)
        i = np.unravel_index(np.abs(t).reshape(t.shape[:-2] + (-1,)).argmax(axis=-1), t.shape[-2:])
        p, q = t[indices + i[:1]], t[indices + (slice(None), i[1])]

        if self.dim > 2 and not np.all(
            is_multiple(
                q[..., None] * p[..., None, :],
                t,
                rtol=EQ_TOL_REL,
                atol=EQ_TOL_ABS,
                axis=(-2, -1),
            )
        ):
            raise NotReducible("Quadric has no decomposition in 2 components.")

        # TODO: make order of components reproducible

        p, q = np.real_if_close(p), np.real_if_close(q)

        if self.is_dual:
            return [PointCollection(p, copy=False), PointCollection(q, copy=False)]
        elif n == 3:
            return [LineCollection(p, copy=False), LineCollection(q, copy=False)]
        return [PlaneCollection(p, copy=False), PlaneCollection(q, copy=False)]

    def intersect(self, other):
        """Calculates points of intersection of a line or a collection of lines with the quadrics.

        Parameters
        ----------
        other: Line or LineCollection
            The line or lines to intersect the quadrics with.

        Returns
        -------
        list of PointCollection
            The points of intersection

        """
        if isinstance(other, (Line, LineCollection)):
            reducible = np.all(self.is_degenerate)
            if reducible:
                try:
                    e, f = self.components
                except NotReducible:
                    reducible = False

            if not reducible:
                if self.dim > 2:
                    arr = other.array.reshape(other.shape[: -other.tensor_shape[1]] + (-1, self.dim + 1))

                    if isinstance(other, Line):
                        i = arr.nonzero()[0][0]
                        m = Plane(arr[i], copy=False).basis_matrix
                    else:
                        i = np.any(arr, axis=-1).argmax(-1)
                        m = PlaneCollection(arr[tuple(np.indices(i.shape)) + (i,)], copy=False).basis_matrix

                    line = other._matrix_transform(m)

                    q = QuadricCollection(matmul(matmul(m, self.array), m, transpose_b=True), copy=False)
                    return [
                        PointCollection(np.squeeze(matmul(np.expand_dims(p.array, -2), m), -2), copy=False)
                        for p in q.intersect(line)
                    ]
                else:
                    m = hat_matrix(other.array)
                    b = matmul(matmul(m, self.array, transpose_a=True), m)
                    p, q = QuadricCollection(b, is_dual=not self.is_dual, copy=False).components
            else:
                if self.is_dual:
                    p, q = e.join(other), f.join(other)
                else:
                    p, q = e.meet(other), f.meet(other)

            return [p, q]

    @property
    def dual(self):
        """QuadricCollection: The dual quadrics of the quadrics in the collection."""
        return QuadricCollection(inv(self.array), is_dual=not self.is_dual, copy=False)
