import numpy as np
import itertools
import sympy
from .point import Point, Line
from .utils import polyval
from numpy.polynomial import polynomial as pl


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

        c = np.zeros([poly.degree() + 1]*len(symbols))

        indices = [range(poly.degree() + 1)]*len(symbols)
        for idx in itertools.product(*indices):
            x = 1
            for i, p in enumerate(idx):
                x *= symbols[i] ** p
            c[idx] = poly.coeff_monomial(x)

        self.coeff = c

    @property
    def polynomial(self):
        f = 0
        indices = [range(i) for i in self.coeff.shape]
        for index in itertools.product(*indices):
            x = 1
            for i, s in enumerate(self.symbols):
                x *= s**index[i]
            f += self.coeff[index] * x
        return f

    def tangent(self, at: Point):
        dx = polyval(at.array, pl.polyder(self.coeff, axis=0))
        dy = polyval(at.array, pl.polyder(self.coeff, axis=1))
        dz = polyval(at.array, pl.polyder(self.coeff, axis=2))
        return Line(dx, dy, dz)

    def contains(self, pt: Point):
        return np.isclose(polyval(pt.array, self.coeff), 0)


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
        if not np.isclose(np.linalg.det(self.array), 0):
            return [self]
        a = np.sqrt(-np.linalg.det(self.array[np.array([1,2])[:,np.newaxis],np.array([1,2])]))
        b = np.sqrt(-np.linalg.det(self.array[np.array([0,2])[:, np.newaxis], np.array([0,2])]))
        c = np.sqrt(-np.linalg.det(self.array[np.array([0,1])[:, np.newaxis], np.array([0,1])]))
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
