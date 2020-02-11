import math
import warnings
from itertools import combinations

import sympy
import numpy as np
from numpy.polynomial import polynomial as pl
from numpy.lib.scimath import sqrt as csqrt

from .point import Point, Line, Plane, I, J, infty_plane
from .transformation import rotation, translation
from .base import ProjectiveElement, Tensor, _symbols, EQ_TOL_REL, EQ_TOL_ABS
from .exceptions import NotReducible
from .utils import polyval, np_array_to_poly, poly_to_np_array, hat_matrix, is_multiple, adjugate


class AlgebraicCurve(ProjectiveElement):
    """A plane algebraic curve, defined by the zero set of a homogeneous polynomial in 3 variables.

    .. deprecated:: 0.2.1
          The class `AlgebraicCurve` will be removed in geometer 0.3, use the differential geometry
          module of sympy or another library instead.

    Parameters
    ----------
    poly : sympy.Expr or numpy.ndarray
        The polynomial defining the curve. It is automatically homogenized.
    symbols : iterable of sympy.Symbol, optional
        The symbols that are used in the polynomial. By default the symbols (x0, x1, x2) will be used.

    Attributes
    ----------
    symbols : tuple of sympy.Symbol
        The symbols used in the polynomial defining the curve.

    """

    def __init__(self, poly, symbols=None):
        warnings.warn('The class AlgebraicCurve is deprecated, use sympy or other package', category=DeprecationWarning)

        if isinstance(poly, np.ndarray):
            if poly.ndim != 3:
                raise ValueError("Expected a polynomial in 3 variables.")

            self.symbols = _symbols(3) if symbols is None else tuple(symbols)
            super(AlgebraicCurve, self).__init__(poly, covariant=False)
            return

        if not isinstance(poly, sympy.Expr):
            raise ValueError("poly must be ndarray or sympy expression.")

        if symbols is None:
            symbols = poly.free_symbols

        poly = sympy.poly(poly, *symbols)

        self.symbols = symbols

        poly = poly.homogenize(symbols[-1])

        super(AlgebraicCurve, self).__init__(poly_to_np_array(poly, symbols), covariant=False)

    @property
    def polynomial(self):
        """sympy.Poly: The polynomial defining this curve."""
        return np_array_to_poly(self.array, self.symbols)

    def tangent(self, at):
        """Calculates the tangent of the curve at a given point.

        Parameters
        ----------
        at : Point
            The point to calculate the tangent line at.

        Returns
        -------
        Line
            The tangent line.

        """
        dx = [polyval(at.array, pl.polyder(self.array, axis=i)) for i in range(self.dim + 1)]
        return Line(dx)

    @property
    def degree(self):
        """int: The degree of the curve, i.e. the homogeneous order of the defining polynomial."""
        return self.polynomial.homogeneous_order()

    def is_tangent(self, line):
        """Tests if a given line is tangent to the curve.

        The method compares the number of intersections to the degree of the algebraic curve. If the line is tangent
        to the curve it will have at least a double intersection and using Bezout's theorem we know that otherwise the
        number of intersections (counted without multiplicity) is equal to the degree of the curve.

        Parameters
        ----------
        line : Line
            The line to test.

        Returns
        -------
        bool
            True if the given line is tangent to the algebraic curve.

        """
        return len(self.intersect(line)) < self.degree

    def contains(self, pt, tol=1e-8):
        """Tests if a given point lies on the algebraic curve.

        Parameters
        ----------
        pt : Point
            The point to test.
        tol : float, optional
            The accepted tolerance.

        Returns
        -------
        bool
            True if the curve contains the point.

        """
        return np.isclose(polyval(pt.array, self.array), 0, atol=tol)

    def intersect(self, other):
        """Calculates points of intersection with the algebraic curve.

        Parameters
        ----------
        other : Line or AlgebraicCurve
            The object to intersect this curve with.

        Returns
        -------
        list of Point
            The points of intersection.

        """
        sol = set()

        if isinstance(other, Line):
            polys = [self.polynomial] + other.polynomials(self.symbols)

        elif isinstance(other, AlgebraicCurve):
            polys = [self.polynomial, other.polynomial]

        else:
            raise NotImplementedError("Intersection for objects of type %s not supported." % str(type(other)))

        for z in [0, 1]:
            p = [f.subs(self.symbols[-1], z) for f in polys]

            try:
                x = sympy.solve_poly_system(p, *self.symbols[:-1])
                sol.update(tuple(complex(x) for x in cor) + (z,) for cor in x)
            except NotImplementedError:
                continue

        return [Point(np.real_if_close(x)) for x in sol if Tensor(x) != 0]
    
    
class Quadric(ProjectiveElement):
    """Represents a quadric, i.e. the zero set of a polynomial of degree 2, in any dimension.

    The quadric is defined by a symmetric matrix of size n+1 where n is the dimension of the space.

    Parameters
    ----------
    matrix : array_like or Tensor
        A two-dimensional array defining the (n+1)x(n+1) symmetric matrix of the quadric.
    is_dual : bool, optional
        If true, the quadric represents a dual quadric, i.e. all hyperplanes tangent to the non-dual quadric.

    Attributes
    ----------
    symbols : tuple of sympy.Symbol
        The symbols used in the polynomial defining the hypersurface.

    """

    def __init__(self, matrix, is_dual=False):
        self.is_dual = is_dual
        matrix = matrix.array if isinstance(matrix, Tensor) else np.array(matrix)
        self.symbols = _symbols(matrix.shape[0])
        super(Quadric, self).__init__(matrix, covariant=False)

    def __apply__(self, transformation):
        inv = transformation.inverse().array
        result = self.copy()
        result.array = inv.T @ self.array @ inv
        return result

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
        return cls(m + m.T)

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
        return Plane(self.array.dot(at.array))

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

    def contains(self, other, tol=1e-8):
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
    def polynomial(self):
        """sympy.Poly: The polynomial defining this quadric."""
        warnings.warn('The property Quadric.polynomial is deprecated', category=DeprecationWarning)

        return sympy.poly(self.array.dot(self.symbols).dot(self.symbols), self.symbols)

    @property
    def is_degenerate(self):
        """bool: True if the quadric is degenerate."""
        return np.isclose(np.linalg.det(self.array), 0, atol=EQ_TOL_ABS)

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
                p.append(csqrt(-np.linalg.det(self.array[row_ind, col_ind])))

        # use the skew symmetric matrix m to get a matrix of rank 1 defining the same quadric
        m = hat_matrix(p)
        t = self.array + m

        # components are in the non-zero rows and columns (up to scalar multiple)
        i = np.unravel_index(np.abs(t).argmax(), t.shape)
        p, q = t[i[0]], t[:, i[1]]

        if self.dim > 2 and not is_multiple(np.outer(q, p), t, rtol=EQ_TOL_REL, atol=EQ_TOL_ABS):
            raise NotReducible("Quadric has no decomposition in 2 components.")

        if self.is_dual:
            return [Point(p), Point(q)]
        elif n == 3:
            return [Line(p), Line(q)]
        return [Plane(p), Plane(q)]

    def intersect(self, other):
        """Calculates points of intersection of a line with the quadric.

        Parameters
        ----------
        other: Line
            The line to intersect this quadric with.

        Returns
        -------
        list of Point
            The points of intersection

        """
        if isinstance(other, Line):
            reducible = self.is_degenerate
            if reducible:
                try:
                    e, f = self.components
                except NotReducible:
                    reducible = False

            if not reducible:
                if self.dim > 2:
                    arr = other.array.reshape((-1, self.dim + 1))
                    i = np.where(arr != 0)[0][0]
                    m = Plane(arr[i]).basis_matrix
                    q = Quadric(m.dot(self.array).dot(m.T))
                    line_base = other.basis_matrix.T
                    line = Line(*[Point(x) for x in m.dot(line_base).T])
                    return [Point(m.T.dot(p.array)) for p in q.intersect(line)]
                else:
                    m = hat_matrix(other.array)
                    b = m.T.dot(self.array).dot(m)
                    p, q = Conic(b, is_dual=not self.is_dual).components
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
        return Quadric(np.linalg.inv(self.array), is_dual=not self.is_dual)


class Conic(Quadric):
    """A two-dimensional conic.
    """

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
        a, b, c, d, e = a.normalized_array, b.normalized_array, c.normalized_array, d.normalized_array, e.normalized_array
        ace = np.linalg.det([a, c, e])
        bde = np.linalg.det([b, d, e])
        ade = np.linalg.det([a, d, e])
        bce = np.linalg.det([b, c, e])
        m = ace*bde*np.outer(np.cross(a, d), np.cross(b, c)) - ade*bce*np.outer(np.cross(a, c), np.cross(b, d))
        return cls(m+m.T)

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
        return cls(m + m.T)

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

        a2b1 = np.linalg.det([o, a2, b1])
        a2b2 = np.linalg.det([o, a2, b2])
        a1b1 = np.linalg.det([o, a1, b1])
        a1b2 = np.linalg.det([o, a1, b2])

        c1 = csqrt(a2b1*a2b2)
        c2 = csqrt(a1b1*a1b2)

        x = Point(c1 * a1 + c2 * a2)
        y = Point(c1 * a1 - c2 * a2)

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
        t1, t2, t3, t4 = Line(f1, I), Line(f1, J), Line(f2, I), Line(f2, J)
        c = cls.from_tangent(Line(bound.array), Point(t1.array), Point(t2.array), Point(t3.array), Point(t4.array))
        return cls(np.linalg.inv(c.array))

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

        return cls(matrix + matrix.T)

    def intersect(self, other):
        """Calculates points of intersection with the conic.

        Parameters
        ----------
        other: Line or Conic
            The object to intersect this conic with.

        Returns
        -------
        list of Point
            The points of intersection.

        """
        if isinstance(other, Conic):
            if other.is_degenerate:
                g, h = other.components
            else:
                a1, a2, a3 = self.array
                b1, b2, b3 = other.array
                alpha = np.linalg.det(self.array)
                beta = np.linalg.det([a1, a2, b3]) + np.linalg.det([a1, b2, a3]) + np.linalg.det([b1, a2, a3])
                gamma = np.linalg.det([a1, b2, b3]) + np.linalg.det([b1, a2, b3]) + np.linalg.det([b1, b2, a3])
                delta = np.linalg.det(other.array)

                roots = np.roots([alpha, beta, gamma, delta])
                c = Conic(self.array + roots[0] * other.array, is_dual=self.is_dual)
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
        return Line(self.array.dot(pt.array))

    @property
    def foci(self):
        """tuple of Point: The foci of the conic."""
        # Algorithm from Perspectives on Projective Geometry, Section 19.4
        i = self.tangent(at=I)
        j = self.tangent(at=J)

        if isinstance(i, Line) and isinstance(j, Line):
            return i.meet(j),

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

    """

    def __init__(self, center=Point(0, 0), hradius=1, vradius=1):
        r = np.array([vradius ** 2, hradius ** 2, 1])
        c = -center.normalized_array
        d = c * r
        m = np.eye(3, dtype=d.dtype)
        m[[0, 1], [0, 1]] = r[:2]
        m[2, :] = d
        m[:, 2] = d
        m[2, 2] = d.dot(c) - (r[0] * r[1] + 1)
        super(Ellipse, self).__init__(m)


class Circle(Ellipse):
    """A circle in 2D.

    Parameters
    ----------
    center : Point, optional
        The center point of the circle, default is Point(0, 0).
    radius : float, optional
        The radius of the circle, default is 1.

    """

    def __init__(self, center=Point(0, 0), radius=1):
        super(Circle, self).__init__(center, radius, radius)

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
        x = m[0]**2 + m[1]**2 - self.radius**2
        return Point([(1 + x)/2, (1 - x)/2, m[0], m[1], self.radius])

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
        return 2*np.pi*self.radius**2


class Sphere(Quadric):
    """A sphere in any dimension.

    Parameters
    ----------
    center : Point, optional
        The center of the sphere, default is Point(0, 0, 0).
    radius : float, optional
        The radius of the sphere, default is 1.

    """

    def __init__(self, center=Point(0, 0, 0), radius=1):
        c = -center.normalized_array
        m = np.eye(center.shape[0], dtype=np.find_common_type([c.dtype, type(radius)], []))
        m[-1, :] = c
        m[:, -1] = c
        m[-1, -1] = c[:-1].dot(c[:-1])-radius**2
        super(Sphere, self).__init__(m)

    @property
    def center(self):
        """Point: The center of the sphere."""
        return Point(np.append(-self.array[:-1, -1], [self.array[0, 0]]))

    @property
    def radius(self):
        """float: The radius of the sphere."""
        c = self.array[:-1, -1] / self.array[0, 0]
        return np.sqrt(c.dot(c) - self.array[-1, -1] / self.array[0, 0])

    @staticmethod
    def _alpha(n):
        return math.pi**(n/2) / math.gamma(n/2 + 1)

    @property
    def volume(self):
        """float: The volume of the sphere."""
        n = self.dim
        return self._alpha(n)*self.radius**n

    @property
    def area(self):
        """float: The surface area of the sphere."""
        n = self.dim
        return n*self._alpha(n)*self.radius**(n-1)


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

    """

    def __init__(self, vertex=Point(0, 0, 0), base_center=Point(0, 0, 1), radius=1):
        from .operators import dist, angle

        h = dist(vertex, base_center)
        c = (radius / h)**2

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
            m[3, 3] = v[:2].dot(v[:2]) - (radius**2 if np.isinf(h) else v[2]**2 * c)

        # rotate the axis of the cone
        axis = Line(Point(v), Point(v)+Point(0, 0, 1))
        new_axis = Line(vertex, base_center)

        if new_axis != axis:
            a = angle(axis, new_axis)
            e = axis.join(new_axis)
            t = rotation(a, axis=Point(*e.array[:3]))
            t = translation(Point(v)) * t * translation(-Point(v))
            m = t.array.T.dot(m).dot(t.array)

        super(Cone, self).__init__(m)


class Cylinder(Cone):
    """A circular cylinder in 3D.

    Parameters
    ----------
    center : Point, optional
        The center of the cylinder. Default is (0, 0, 0).
    direction : Point
        The direction of the axis of the cylinder. Default is (0, 0, 1).
    radius : float, optional
        The radius of the cylinder. Default is 1.

    """

    def __init__(self, center=Point(0, 0, 0), direction=Point(0, 0, 1), radius=1):
        vertex = infty_plane.meet(Line(center, center+direction))
        super(Cylinder, self).__init__(vertex, center, radius)
