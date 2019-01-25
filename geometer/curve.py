import numpy as np
import sympy
from .point import Point, Line, Plane, I, J
from .base import ProjectiveElement
from .utils.polynomial import polyval, np_array_to_poly, poly_to_np_array
from numpy.polynomial import polynomial as pl
from numpy.lib.scimath import sqrt as csqrt


class AlgebraicCurve(ProjectiveElement):
    """A general algebraic curve, defined by the zero set of a homogeneous polynomial.

    Parameters
    ----------
    poly : :obj:`sympy.Expr` or :obj:`numpy.ndarray`
        The polynomial defining the curve.
    symbols : :obj:`tuple` of :obj:`sympy.Symbol`, optional
        The symbols that are used in the polynomial. By default the symbols (x1, ..., xn) will be used.

    Attributes
    ----------
    symbols : :obj:`tuple` of :obj:`sympy.Symbol`
        The symbols used in the polynomial defining the curve.

    """

    def __init__(self, poly, symbols=None):
        if symbols is None:
            symbols = sympy.symbols(["x" + str(i) for i in range(len(poly.shape))])

        if isinstance(poly, np.ndarray):
            self.symbols = symbols
            super(AlgebraicCurve, self).__init__(poly)
            return

        if not isinstance(poly, sympy.Expr):
            raise ValueError("poly must be ndarray or sympy expression")

        if isinstance(poly, sympy.Poly):
            symbols = poly.free_symbols
        else:
            poly = sympy.poly(poly, *symbols)

        self.symbols = symbols

        poly = poly.homogenize(symbols[-1])

        super(AlgebraicCurve, self).__init__(poly_to_np_array(poly, symbols), contravariant_indices=[0, 1])

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
        return np.isclose(float(polyval(pt.array, self.array)), 0)

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
            for z in [0, 1]:
                polys = [self.polynomial.subs(self.symbols[-1], z)]
                for f in other.polynomials(self.symbols):
                    polys.append(f.subs(self.symbols[-1], z))

                try:
                    x = sympy.solve_poly_system(polys, *self.symbols[:-1])
                    sol.update(tuple(float(x) for x in cor) + (z,) for cor in x)
                except NotImplementedError:
                    continue

        if isinstance(other, AlgebraicCurve):
            for z in [0, 1]:
                f = self.polynomial.subs(self.symbols[-1], z)
                g = other.polynomial.subs(self.symbols[-1], z)

                try:
                    x = sympy.solve_poly_system([f, g], *self.symbols[:-1])
                    sol.update(tuple(float(x) for x in cor) + (z,) for cor in x)
                except NotImplementedError:
                    continue

        if (0, 0, 0) in sol:
            sol.remove((0, 0, 0))

        return [Point(p) for p in sol]
    
    
class Quadric(AlgebraicCurve):
    """Represents a quadric, i.e. the zero set of a polynomial of degree 2, in any dimension.

    The quadric is defined by a matrix of size n+1 where n is the dimension of the space.

    Parameters
    ----------
    matrix : array_like
        The array defining a (n+1)x(n+1) matrix.

    """

    def __init__(self, matrix):
        self.matrix = np.array(matrix)
        symbols = sympy.symbols(["x" + str(i) for i in range(self.matrix.shape[1])])
        super(Quadric, self).__init__(self.matrix.dot(symbols).dot(symbols), symbols=symbols)

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
        return Plane(self.matrix.dot(at.array))

    @property
    def is_degenerate(self):
        """bool: True if the quadric is degenerate."""
        return np.isclose(float(np.linalg.det(self.matrix)), 0)


class Conic(Quadric):
    """A two-dimensional conic.

    Parameters
    ----------
    matrix : array_like
        A two dimensional array that defines the 2x2 matrix of the conic.
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
        ace = np.linalg.det([a.array, c.array, e.array])
        bde = np.linalg.det([b.array, d.array, e.array])
        ade = np.linalg.det([a.array, d.array, e.array])
        bce = np.linalg.det([b.array, c.array, e.array])
        m = ace*bde*np.outer(np.cross(a.array, d.array), np.cross(b.array, c.array))\
            - ade*bce*np.outer(np.cross(a.array, c.array), np.cross(b.array, d.array))
        return Conic(m+m.T)

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

    @property
    def components(self):
        """:obj:`list` of :obj:`ProjectiveElement`: The components of a degenerate conic."""
        if not self.is_degenerate:
            return [self]
        a = csqrt(-np.linalg.det(self.matrix[np.array([1,2])[:, np.newaxis], np.array([1,2])]))
        b = csqrt(-np.linalg.det(self.matrix[np.array([0,2])[:, np.newaxis], np.array([0,2])]))
        c = csqrt(-np.linalg.det(self.matrix[np.array([0,1])[:, np.newaxis], np.array([0,1])]))
        m = np.array([[0, c, -b],
                      [-c, 0, a],
                      [b, -a, 0]])
        t = self.matrix + m
        x, y = np.nonzero(t)
        result = []
        for a in np.concatenate((t[x,:], t[:,y].T)):
            l = Point(a) if self.is_dual else Line(a)
            if l not in result:
                result.append(l)
        return result

    def intersect(self, other):
        """Calculates points of intersection with the conic.

        Parameters
        ----------
        other: :obj:`Line` or :obj:`Conic`
            The object to intersect this curve with.

        Returns
        -------

        """
        if isinstance(other, Line):
            x, y, z = tuple(other.array)
            m = np.array([[0, z, -y],
                          [-z, 0, x],
                          [y, -x, 0]])
            b = m.T.dot(self.matrix).dot(m)
            return Conic(b, is_dual=True).components
        if isinstance(other, Conic):
            if other.is_degenerate:
                if self.is_degenerate:
                    results = []
                    for g in self.components:
                        for h in other.components:
                            if g != h:
                                results.append(g.meet(h))
                    return results
                results = []
                for l in other.components:
                    for i in self.intersect(l):
                        if i not in results:
                            results.append(i)
                return results
            x = sympy.symbols("x")
            m = sympy.Matrix(self.matrix + x * other.matrix)
            f = sympy.poly(m.det(), x)
            roots = np.roots(f.coeffs())
            c = Conic(self.matrix + roots[0]*other.matrix)
            results = []
            for l in c.components:
                for i in self.intersect(l):
                    if i not in results:
                        results.append(i)
            return results
        return super(Conic, self).intersect(other)

    def tangent(self, at):
        """Calculates the tangent line of the conic at a given point.

        If the given point doesn't lie on the conic, the resulting line will be the polar line.

        Parameters
        ----------
        at : Point
            The point to calculate the tangent at.

        Returns
        -------
        Line
            The tangent line (or polar if the point doesn't lie on the conic).

        """
        return Line(self.matrix.dot(at.array))

    @property
    def dual(self):
        """Conic: The dual conic."""
        return Conic(np.linalg.inv(self.matrix), is_dual=not self.is_dual)


absolute_conic = Conic(np.eye(3))


class Circle(Conic):
    """A circle in 2D.

    Parameters
    ----------
    center : Point
        The center point of the circle.
    radius : float
        The radius of the circle.

    """

    def __init__(self, center=Point(0, 0), radius=1):
        super(Circle, self).__init__([[1, 0, -center.array[0]],
                                      [0, 1, -center.array[1]],
                                      [-center.array[0], -center.array[1], center.array[:-1].dot(center.array[:-1])-radius**2]])

    @property
    def center(self):
        """Point: the center point of the circle."""
        l = self.tangent(at=I)
        m = self.tangent(at=J)
        return l.meet(m)
