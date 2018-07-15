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