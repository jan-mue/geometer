import itertools
import warnings

import sympy
import numpy as np
from numpy.polynomial import polynomial as pl


def polyval(x, c):
    """Evaluate a multivariate polynomial at a specified point.

    .. deprecated:: 0.2.1
          `polyval` will be removed in geometer 0.3.

    If `x` is an array of length :math:`n` and `c` an array of shape :math:`(m, \ldots, m)`, the result is

    .. math::

        \sum_{0\leq i_1, \ldots, i_n \leq m-1} c[i_1, \ldots, i_n] x[1]^{i_1}\cdots x[n]^{i_n}.

    Parameters
    ----------
    x : array_like
        The point to evaluate the polynomial at.
    c : array_like
        Array of coefficients where the different axis correspond to different variables with the degree
        in each axis given by the corresponding index.

    Returns
    -------
    :obj:`float` or :obj:`complex`
        The result of the evaluation as described above.

    """
    warnings.warn('The function polyval is deprecated', category=DeprecationWarning)

    if len(x) != c.ndim:
        raise ValueError("Dimension of point and polynomial do not match.")
    for xi in x:
        c = pl.polyval(xi, c, tensor=False)
    return c


def poly_to_np_array(p, symbols):
    """Convert a sympy polynomial to an array of coefficients as required by numpy.

    .. deprecated:: 0.2.1
          `poly_to_np_array` will be removed in geometer 0.3.

    If `p` is a multivariate polynomial with the variables :math:`x_1, \ldots, x_n` specified in `symbols`, the
    resulting array will be such that

    .. math::

        p = \sum_{0\leq i_1, \ldots, i_n \leq m-1} c[i_1, \ldots, i_n] x_1^{i_1}\cdots x_n^{i_n}.

    Parameters
    ----------
    p : sympy.Expr
        The sympy expression that represents the polynomial.
    symbols : list of sympy.Symbol
        The variables used in the polynomial expression.

    Returns
    -------
    numpy.ndarray
        The coefficients of the polynomial as described above.

    """
    warnings.warn('The function poly_to_np_array is deprecated', category=DeprecationWarning)

    p = sympy.poly(p, symbols)
    c = np.zeros([p.total_degree() + 1] * len(symbols), dtype=complex)

    indices = [range(p.total_degree() + 1)] * len(symbols)
    for idx in itertools.product(*indices):
        x = 1
        for i, a in enumerate(idx):
            x *= symbols[i] ** a
        c[idx] = p.coeff_monomial(x)

    return np.real_if_close(c)


def np_array_to_poly(c, symbols):
    """Converts an array of coefficients into a sympy polynomial.

    .. deprecated:: 0.2.1
          `np_array_to_poly` will be removed in geometer 0.3.

    If `c` an array of shape :math:`(m, \ldots, m)` and variables :math:`x_1, \ldots, x_n` are given as `symbols`, the
    resulting polynomial will be given by

    .. math::

        \sum_{0\leq i_1, \ldots, i_n \leq m-1} c[i_1, \ldots, i_n] x_1^{i_1}\cdots x_n^{i_n}.

    Parameters
    ----------
    c : array_like
        Array of coefficients where the different axis correspond to different variables with the degree
        in each axis given by the corresponding index.
    symbols : list of sympy.Symbol
        The symbols to be used as variables of the polynomial.

    Returns
    -------
    sympy.Poly
        The resulting polynomial as described above.

    """
    warnings.warn('The function np_array_to_poly is deprecated', category=DeprecationWarning)

    c = np.array(c)
    f = 0
    indices = [range(i) for i in c.shape]
    for index in itertools.product(*indices):
        x = 1
        for i, s in enumerate(symbols):
            x *= s ** index[i]
        a = c[index]
        f += a * x
    return sympy.poly(f, symbols)
