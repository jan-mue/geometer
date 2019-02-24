import numpy as np
from sympy.abc import x, y, z
from geometer import Point, Line, Conic, Circle, Quadric, Plane
from geometer.curve import AlgebraicCurve


class TestAlgebraicCurve:

    def test_polynomial(self):
        coeff = np.array([[[0, 1], [1, 0]], [[1, 0], [0, 0]]])
        curve = AlgebraicCurve(coeff, symbols=[x, y, z])
        assert curve.polynomial == x + y + z

    def test_contains(self):
        coeff = np.array([[[0, 0], [1, 0]], [[1, 0], [0, 0]]])
        curve = AlgebraicCurve(coeff)
        assert curve.contains(Point(0, 0))

    def test_tangent(self):
        curve = AlgebraicCurve(x ** 2 + y ** 2 - 1, symbols=[x, y, z])
        assert curve.tangent(at=Point(1, 0)) == Line(1, 0, 0) + Point(1, 0)
        assert curve.is_tangent(Line(1, 0, 0) + Point(1, 0))

    def test_intersect(self):
        c1 = AlgebraicCurve(x ** 2 + y ** 2 - 1, symbols=[x, y, z])
        c2 = AlgebraicCurve(y, symbols=[x, y, z])
        i = c1.intersect(c2)
        assert len(i) == 2
        assert Point(1, 0) in i
        assert Point(-1, 0) in i


class TestConic:

    def test_from_points(self):
        a = Point(0, 1)
        b = Point(0, -1)
        c = Point(1.5, 0.5)
        d = Point(1.5, -0.5)
        e = Point(-1.5, 0.5)

        conic = Conic.from_points(a, b, c, d, e)

        assert conic.contains(a)
        assert conic.contains(b)
        assert conic.contains(c)
        assert conic.contains(d)
        assert conic.contains(e)

    def test_intersections(self):
        c = Circle(Point(0, 0), 1)
        i = c.intersect(Line(0, 1, 0))
        assert len(i) == 2
        assert Point(1, 0) in i
        assert Point(-1, 0) in i

        c2 = Circle(Point(0, 2), 1)
        assert Point(0, 1) in c.intersect(c2)

    def test_contains(self):
        c = Conic([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, -1]])

        assert c.contains(Point(1, 0))

    def test_circle(self):
        c = Circle(Point(0, 1), 1)
        assert c.contains(Point(0, 2))
        assert c.contains(Point(1, 1))
        assert c.tangent(at=Point(0, 0)) == Line(0, 1, 0)
        assert c.center == Point(0, 1)

    def test_quadric(self):
        q = Quadric([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, -1]])

        assert q.contains(Point(1, 0, 0))
        assert q.tangent(at=Point(1, 0, 0)) == Plane(Point(1, 0, 0), Point(1, 0, 1), Point(1, 1, 0))
