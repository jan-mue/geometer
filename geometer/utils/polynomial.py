import itertools

import sympy
import numpy as np
from numpy.polynomial import polynomial as pl


def polyval(x, c):
    if len(x) != len(c.shape):
        raise ValueError("Dimension of point and polynomial do not match.")
    for xi in x:
        c = pl.polyval(xi, c, tensor=False)
    return c


def poly_to_np_array(p, symbols):
    c = np.zeros([p.total_degree() + 1] * len(symbols), dtype=object)

    indices = [range(p.total_degree() + 1)] * len(symbols)
    for idx in itertools.product(*indices):
        x = 1
        for i, a in enumerate(idx):
            x *= symbols[i] ** a
        c[idx] = p.coeff_monomial(x)

    return c


def np_array_to_poly(c, symbols):
    f = 0
    indices = [range(i) for i in c.shape]
    for index in itertools.product(*indices):
        x = 1
        for i, s in enumerate(symbols):
            x *= s ** index[i]
        a = c[index]
        f += a * x
    return sympy.Poly(f, symbols)
