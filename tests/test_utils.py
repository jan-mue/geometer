from geometer.utils import *
import numpy as np
from fractions import Fraction
from sympy.abc import x, y, z


def test_poly_to_array():
    p = 5*x**2 + 2*x*y + 3*y**2 + z**3
    a = poly_to_np_array(p, (x, y, z))
    assert np.array_equal(a, [[[0, 0, 0, 1], [0, 0, 0, 0], [3, 0, 0, 0], [0, 0, 0, 0]],
                              [[0, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                              [[5, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                              [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])


def test_array_to_poly():
    a = [[[0, 0, 0, 1], [0, 0, 0, 0], [3, 0, 0, 0], [0, 0, 0, 0]],
         [[0, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
         [[5, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
         [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
    assert np_array_to_poly(a, (x, y, z)) == 5*x**2 + 2*x*y + 3*y**2 + z**3


def test_null_space():
    a = np.array([[2, Fraction(1, 3), 4, Fraction(1, 3)],
                  [5, Fraction(3, 3), 7, Fraction(1, 6)],
                  [4, Fraction(7, 3), 8, Fraction(1, 7)],
                  [2, Fraction(1, 3), 4, Fraction(1, 3)]])
    assert np.all(abs(a.dot(null_space(a))) < 1e-9)
