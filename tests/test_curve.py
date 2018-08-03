import numpy as np
from sympy import symbols
from geometer import Point, Line, Conic, Circle
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

def test_from_points():
    a = Point(-1, 0)
    b = Point(0, 3)
    c = Point(1, 2)
    d = Point(2, 1)
    e = Point(0, -1)

    conic = Conic.from_points(a,b,c,d,e)

    assert conic.contains(a)
    assert conic.contains(b)
    assert conic.contains(c)
    assert conic.contains(d)
    assert conic.contains(e)

def test_circle():
    c = Circle(Point(0,1), 1)
    assert c.contains(Point(0,2))
    assert c.contains(Point(1,1))
    assert c.tangent(at=Point(0,0)) == Line(0,1,0)

def test_conic():
    c = Conic([[1,0,0],
               [0,1,0],
               [0,0,-1]])

    assert c.contains(Point(1,0))
