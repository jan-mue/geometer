import numpy as np
from sympy import symbols
from geometer import Point, Line
from geometer.curve import AlgebraicCurve


def test_polynomial():
    coeff = np.array([[[0,1], [1,0]], [[1,0],[0,0]]])
    curve = AlgebraicCurve(coeff)
    x0, x1, x2 = symbols("x0 x1 x2")
    assert curve.polynomial == x0 + x1 + x2

def test_contains():
    coeff = np.array([[[0,0], [1,0]], [[1,0],[0,0]]])
    curve = AlgebraicCurve(coeff)
    assert curve.contains(Point(0,0))

def test_tangent():
    x, y, z = symbols("x y z")
    curve = AlgebraicCurve(x**2 + y**2 - 1, symbols=[x, y, z])
    assert curve.tangent(at=Point(1,0)) == Line(1,0,0)
