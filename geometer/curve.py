import numpy as np
import itertools
import sympy
from .point import Point, Line
from .utils import polyval
from numpy.polynomial import polynomial as pl


class AlgebraicCurve:

    def __init__(self, poly, symbols=None, origin=None):
        if symbols is None:
            symbols = sympy.symbols(["x" + str(i) for i in range(len(poly.shape))])

        self.origin = origin or Point(*([0] * (len(symbols) - 1)))

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
        dx = polyval((at - self.origin).array, pl.polyder(self.coeff, axis=0))
        dy = polyval((at - self.origin).array, pl.polyder(self.coeff, axis=1))
        dz = polyval((at - self.origin).array, pl.polyder(self.coeff, axis=2))
        return Line(dx, dy, dz) + self.origin

    def contains(self, pt: Point):
        return np.isclose(polyval((pt - self.origin).array, self.coeff), 0)


class Conic(AlgebraicCurve):

    def __init__(self, matrix, origin=Point(0,0)):
        self.array = np.array(matrix)
        m = sympy.Matrix(matrix)
        x, y, z = sympy.symbols("x y z")
        super(Conic, self).__init__(sum(a*b for a,b in zip(m.dot([x, y, z]), [x, y, z])), symbols=[x,y,z], origin=origin)

    @classmethod
    def from_points(cls, a, b, c, d, e, *, origin=Point(0,0)):
        ace = np.linalg.det([a.array, c.array, e.array])
        bde = np.linalg.det([b.array, d.array, e.array])
        ade = np.linalg.det([a.array, d.array, e.array])
        bce = np.linalg.det([b.array, c.array, e.array])
        m = ace*bde*np.outer(np.cross(a.array, d.array), np.cross(b.array, c.array))\
            - ade*bce*np.outer(np.cross(a.array, c.array), np.cross(b.array, d.array))
        return Conic(m+m.T, origin=origin)

    def tangent(self, at: Point):
        return Line(self.array.dot((at - self.origin).array)) + self.origin


class Circle(Conic):

    def __init__(self, center: Point, radius: float):
        super(Circle, self).__init__([[1,0,0],
                                      [0,1,0],
                                      [0,0,-radius**2]], center)
