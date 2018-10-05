import numpy as np
import sympy
from .point import Point, Line, I, J
from .utils.polynomial import polyval, np_array_to_poly, poly_to_np_array
from numpy.polynomial import polynomial as pl
from numpy.lib.scimath import sqrt as csqrt


class AlgebraicCurve:

    def __init__(self, poly, symbols=None):
        if symbols is None:
            symbols = sympy.symbols(["x" + str(i) for i in range(len(poly.shape))])

        if isinstance(poly, np.ndarray):
            self.coeff = poly
            self.symbols = symbols
            return

        if not isinstance(poly, sympy.Expr):
            raise ValueError("poly must be ndarray or sympy expression")

        if isinstance(poly, sympy.Poly):
            symbols = poly.free_symbols
        else:
            poly = sympy.poly(poly, *symbols)

        self.symbols = symbols

        poly = poly.homogenize(symbols[-1])

        self.coeff = poly_to_np_array(poly, symbols)

    @property
    def polynomial(self):
        return np_array_to_poly(self.coeff, self.symbols)

    def tangent(self, at: Point):
        dx = polyval(at.array, pl.polyder(self.coeff, axis=0))
        dy = polyval(at.array, pl.polyder(self.coeff, axis=1))
        dz = polyval(at.array, pl.polyder(self.coeff, axis=2))
        return Line(dx, dy, dz)

    @property
    def degree(self):
        return self.polynomial.homogeneous_order()

    def is_tangent(self, line: Line):
        return len(self.intersections(line)) < self.degree

    def contains(self, pt: Point):
        return np.isclose(float(polyval(pt.array, self.coeff)), 0)

    def intersections(self, other):
        if isinstance(other, Line):
            sol = set()

            for z in [0, 1]:
                f = self.polynomial.subs(self.symbols[-1], z)
                g = other.polynomial.subs(self.symbols[-1], z)
                try:
                    x = sympy.solve_poly_system([f, g], *self.symbols[:-1])
                    sol.update(cor + (z,) for cor in x)
                except NotImplementedError:
                    continue

            if (0,0,0) in sol:
                sol.remove((0,0,0))

            return [Point(p) for p in sol]


class EllipticCurve(AlgebraicCurve):
    """Representation of an Elliptic Curve.

    f(x, y) = y**2 - x**3 - a*x* - b

    """

    o = Point([0,1,0])

    def __init__(self, a, b):
        x,y,z = sympy.symbols("x y z")
        super(EllipticCurve, self).__init__(z*y**2 - x**3 - a*x*z**2 - b*z**3, symbols=(x,y,z))

    def _join_and_intersect(self, a, b):
        if a == b:
            l = self.tangent(at=a)
        else:
            l = a.join(b)

        intersections = self.intersections(l)
        if len(intersections) == 1:
            return intersections[0]
        if len(intersections) == 2:
            for point in intersections:
                if self.tangent(at=point) == l:
                    return point
        if len(intersections) == 3:
            intersections.remove(a)
            intersections.remove(b)
            return intersections[0]

    def add(self, p:Point, q:Point):
        r = self._join_and_intersect(p, q)
        return self.invert(r)

    def invert(self, pt: Point):
        return self._join_and_intersect(self.o, pt)


class Conic(AlgebraicCurve):

    def __init__(self, matrix, is_dual=False):
        self.array = np.array(matrix)
        self.is_dual = is_dual
        m = sympy.Matrix(matrix)
        x, y, z = sympy.symbols("x y z")
        super(Conic, self).__init__(sum(a*b for a,b in zip(m.dot([x, y, z]), [x, y, z])), symbols=[x,y,z])

    @classmethod
    def from_points(cls, a, b, c, d, e):
        ace = np.linalg.det([a.array, c.array, e.array])
        bde = np.linalg.det([b.array, d.array, e.array])
        ade = np.linalg.det([a.array, d.array, e.array])
        bce = np.linalg.det([b.array, c.array, e.array])
        m = ace*bde*np.outer(np.cross(a.array, d.array), np.cross(b.array, c.array))\
            - ade*bce*np.outer(np.cross(a.array, c.array), np.cross(b.array, d.array))
        return Conic(m+m.T)

    @classmethod
    def from_lines(cls, g:Line, h:Line):
        m = np.outer(g.array, h.array)
        return Conic(m + m.T)

    @property
    def components(self):
        if not self.is_degenerate:
            return [self]
        a = csqrt(-np.linalg.det(self.array[np.array([1,2])[:,np.newaxis],np.array([1,2])]))
        b = csqrt(-np.linalg.det(self.array[np.array([0,2])[:, np.newaxis], np.array([0,2])]))
        c = csqrt(-np.linalg.det(self.array[np.array([0,1])[:, np.newaxis], np.array([0,1])]))
        m = np.array([[0, c, -b],
                      [-c, 0, a],
                      [b, -a, 0]])
        t = self.array + m
        x, y = np.nonzero(t)
        result = []
        for a in np.concatenate((t[x,:], t[:,y].T)):
            l = Point(a) if self.is_dual else Line(a)
            if l not in result:
                result.append(l)
        return result

    def intersections(self, other):
        if isinstance(other, Line):
            x,y,z = tuple(other.array)
            m = np.array([[0, z, -y],
                          [-z, 0, x],
                          [y, -x, 0]])
            b = m.T.dot(self.array).dot(m)
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
                    for i in self.intersections(l):
                        if i not in results:
                            results.append(i)
                return results
            x = sympy.symbols("x")
            m = sympy.Matrix(self.array + x * other.array)
            f = sympy.Poly(m.det(), x)
            roots = np.roots(f.coeffs())
            c = Conic(self.array + roots[0]*other.array)
            results = []
            for l in c.components:
                for i in self.intersections(l):
                    if i not in results:
                        results.append(i)
            return results

    @property
    def is_degenerate(self):
        return np.isclose(float(np.linalg.det(self.array)), 0)

    def tangent(self, at: Point):
        return Line(self.array.dot(at.array))

    @property
    def dual(self):
        return Conic(np.linalg.inv(self.array), is_dual=not self.is_dual)


class Circle(Conic):

    def __init__(self, center: Point = Point(0,0), radius: float = 1):
        super(Circle, self).__init__([[1,0,-center.array[0]],
                                      [0,1,-center.array[1]],
                                      [-center.array[0],-center.array[1],center.array[:-1].dot(center.array[:-1])-radius**2]])

    @property
    def center(self):
        l = self.tangent(at=I)
        m = self.tangent(at=J)
        return l.meet(m)
