import numpy as np
import sympy
from .point import Point, Line, Plane, I, J
from .base import ProjectiveElement, Tensor, _symbols
from .utils import polyval, det, inv, np_array_to_poly, poly_to_np_array, isclose
from numpy.polynomial import polynomial as pl
from numpy.lib.scimath import sqrt as csqrt


class AlgebraicCurve(ProjectiveElement):
    """A general algebraic curve, defined by the zero set of a homogeneous polynomial.

    Parameters
    ----------
    poly : :obj:`sympy.Expr` or :obj:`numpy.ndarray`
        The polynomial defining the curve. It is automatically homogenized.
    symbols : :obj:`tuple` of :obj:`sympy.Symbol`, optional
        The symbols that are used in the polynomial. By default the symbols (x1, ..., xn) will be used.

    Attributes
    ----------
    symbols : :obj:`tuple` of :obj:`sympy.Symbol`
        The symbols used in the polynomial defining the curve.

    """

    def __init__(self, poly, symbols=None):

        if isinstance(poly, np.ndarray):
            self.symbols = symbols or _symbols(poly.shape[0])
            super(AlgebraicCurve, self).__init__(poly, covariant=False)
            return

        if not isinstance(poly, sympy.Expr):
            raise ValueError("poly must be ndarray or sympy expression")

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
        """Calculates the tangent space of the curve at a given point.

        Parameters
        ----------
        at : Point
            The point to calculate the tangent space at.

        Returns
        -------
        :obj:`Line` or :obj:`Plane`
            The tangent space.

        """
        dx = [polyval(at.array, pl.polyder(self.array, axis=i)) for i in range(self.dim + 1)]
        if self.dim == 2:
            return Line(dx)
        return Plane(dx)

    @property
    def degree(self):
        """int: The degree of the curve, i.e. the homogeneous order of the defining polynomial."""
        return self.polynomial.homogeneous_order()

    def is_tangent(self, line):
        """Tests if a given line is tangent to the curve.

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

    def contains(self, pt):
        """Tests if a given point lies on the algebraic curve.

        Parameters
        ----------
        pt: Point
            The point to test.

        Returns
        -------
        bool
            True if the curve contains the point.

        """
        return isclose(polyval(pt.array, self.array), 0)

    def intersect(self, other):
        """Calculates points of intersection with the algebraic curve.

        Parameters
        ----------
        other : :obj:`Line` or :obj:`AlgebraicCurve`
            The object to intersect this curve with.

        Returns
        -------
        :obj:`list` of :obj:`Point`
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
    
    
class Quadric(AlgebraicCurve):
    """Represents a quadric, i.e. the zero set of a polynomial of degree 2, in any dimension.

    The quadric is defined by a symmetric matrix of size n+1 where n is the dimension of the space.

    Parameters
    ----------
    matrix : array_like or Tensor
        The array defining a (n+1)x(n+1) symmetric matrix.

    """

    def __init__(self, matrix):
        matrix = matrix.array if isinstance(matrix, Tensor) else np.array(matrix)
        super(Quadric, self).__init__(matrix)

    def tangent(self, at):
        """Returns the hyperplane defining the tangent space at a given point.

        Parameters
        ----------
        at : Point
            The point at which the tangent space is calculated.

        Returns
        -------
        Plane
            The tangent plane at the given point.

        """
        return Plane(self.array.dot(at.array))

    def contains(self, pt):
        """Tests if a given point lies on the quadric.

        Parameters
        ----------
        pt: Point
            The point to test.

        Returns
        -------
        bool
            True if the quadric contains the point.

        """
        return isclose(pt.array.dot(self.array.dot(pt.array)), 0)

    @property
    def polynomial(self):
        """sympy.Poly: The polynomial defining this quadric."""
        return sympy.poly(self.array.dot(self.symbols).dot(self.symbols), self.symbols)

    @property
    def is_degenerate(self):
        """bool: True if the quadric is degenerate."""
        return isclose(det(self.array), 0)


class Conic(Quadric):
    """A two-dimensional conic.

    Parameters
    ----------
    matrix : array_like or Tensor
        A two dimensional array that defines the symmetric 3x3 matrix of the conic.
    is_dual : bool, optional
        If true, the conic represents a dual conic, i.e. all lines tangent to the non-dual conic.

    """

    def __init__(self, matrix, is_dual=False):
        self.is_dual = is_dual
        super(Conic, self).__init__(matrix)

    @classmethod
    def from_points(cls, a, b, c, d, e):
        """Construct a conic through five points.

        Parameters
        ----------
        a : Point
        b : Point
        c : Point
        d : Point
        e : Point

        Returns
        -------
        Conic
            The resulting conic.

        """
        a, b, c, d, e = a.normalized_array, b.normalized_array, c.normalized_array, d.normalized_array, e.normalized_array
        ace = det([a, c, e])
        bde = det([b, d, e])
        ade = det([a, d, e])
        bce = det([b, c, e])
        m = ace*bde*np.outer(np.cross(a, d), np.cross(b, c)) - ade*bce*np.outer(np.cross(a, c), np.cross(b, d))
        return Conic(np.real_if_close(m+m.T))

    @classmethod
    def from_lines(cls, g, h):
        """Construct a degenerate conic from two lines.

        Parameters
        ----------
        g: Line
        h: Line

        Returns
        -------
        Conic
            The resulting conic.

        """
        m = np.outer(g.array, h.array)
        return Conic(m + m.T)

    @classmethod
    def from_points_and_tangent(cls, a, b, c, d, tangent):
        """Construct a conic through four points and tangent to a line.

        Parameters
        ----------
        a : Point
        b : Point
        c : Point
        d : Point
        tangent : Line

        Returns
        -------
        Conic
            The resulting conic.

        """
        if any(tangent.contains(p) for p in [a, b, c, d]):
            raise ValueError("The supplied points cannot lie on the supplied tangent!")

        a1, a2 = Line(a, c).meet(tangent).normalized_array, Line(b, d).meet(tangent).normalized_array
        b1, b2 = Line(a, b).meet(tangent).normalized_array, Line(c, d).meet(tangent).normalized_array

        o = [1, 0, 1]
        i = 0
        while tangent.contains(Point(o)):
            o[2-i] = 1
            i += 1

        a2b1 = det([o, a2, b1])
        a2b2 = det([o, a2, b2])
        a1b1 = det([o, a1, b1])
        a1b2 = det([o, a1, b2])

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
        f1 : Point
            A focal point.
        f2 : Point
            A focal point.
        bound : Point
            A boundary point that lies on the conic.

        Returns
        -------
        Conic
            The resulting conic.

        """
        t1, t2, t3, t4 = Line(f1, I), Line(f1, J), Line(f2, I), Line(f2, J)
        c = cls.from_points_and_tangent(Point(t1.array), Point(t2.array), Point(t3.array), Point(t4.array), Line(bound.array))
        return Conic(inv(c.array))

    @property
    def components(self):
        """:obj:`list` of :obj:`ProjectiveElement`: The components of a degenerate conic."""
        a = csqrt(-det(self.array[[[1], [2]], [1, 2]]))
        b = csqrt(-det(self.array[[[0], [2]], [0, 2]]))
        c = csqrt(-det(self.array[[[0], [1]], [0, 1]]))
        m = np.array([[0, c, -b],
                      [-c, 0, a],
                      [b, -a, 0]])
        t = self.array + m
        i = np.unravel_index(np.abs(t).argmax(), t.shape)
        if self.is_dual:
            return [Point(t[i[0]]), Point(t[:, i[1]])]
        return [Line(t[i[0]]), Line(t[:, i[1]])]

    def intersect(self, other):
        """Calculates points of intersection with the conic.

        Parameters
        ----------
        other: Line or Conic
            The object to intersect this curve with.

        Returns
        -------
        list of Point
            The points of intersection

        """
        if isinstance(other, (Line, Point)):
            if self.is_degenerate:
                g, h = self.components
                if self.is_dual:
                    p, q = g.join(other), h.join(other)
                else:
                    p, q = g.meet(other), h.meet(other)
            else:
                x, y, z = other.array
                m = np.array([[0, z, -y],
                              [-z, 0, x],
                              [y, -x, 0]])
                b = m.T.dot(self.array).dot(m)
                p, q = Conic(b, is_dual=not self.is_dual).components

            if p == q:
                return [p]

            return [p, q]

        if isinstance(other, Conic):
            if other.is_degenerate:
                g, h = other.components

            else:
                x = _symbols(1)
                m = sympy.Matrix(self.array + x * other.array)
                f = sympy.poly(m.det(), x)
                roots = np.roots(f.coeffs())
                c = Conic(self.array + roots[0]*other.array, is_dual=self.is_dual)
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
        i = self.tangent(at=I)
        j = self.tangent(at=J)
        if isinstance(i, Line) and isinstance(j, Line):
            return i.meet(j),
        intersections = [i[0].meet(j[0]), i[1].meet(j[1]), i[0].meet(j[1]), i[1].meet(j[0])]
        return tuple(p for p in intersections if np.all(np.isreal(p.normalized_array)))

    @property
    def dual(self):
        """Conic: The dual conic."""
        return Conic(inv(self.array), is_dual=not self.is_dual)


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
        m = np.diag(r)
        m[2, :] = c * r
        m[:, 2] = c * r
        m[2, 2] = r.dot(c ** 2) - (r[0] * r[1] + 1)
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
        """Point: the center point of the circle."""
        return self.foci[0]


class Sphere(Quadric):

    def __init__(self, center=Point(0, 0, 0), radius=1):
        c = -center.normalized_array
        m = np.eye(4, dtype=(c*radius).dtype)
        m[3, :] = c
        m[:, 3] = c
        m[3, 3] = c[:-1].dot(c[:-1])-radius**2
        super(Sphere, self).__init__(m)
