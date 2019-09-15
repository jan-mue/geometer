import numpy as np
from sympy.abc import x, y, z
from geometer import Point, Line, Conic, Circle, Quadric, Plane, Ellipse, Sphere, Cone, Cylinder, AlgebraicCurve, crossratio


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

    def test_ellipse(self):
        el = Ellipse(Point(1, 2), 2, 3)

        assert el.contains(Point(-1, 2))
        assert el.contains(Point(3, 2))
        assert el.contains(Point(1, 5))
        assert el.contains(Point(1, -1))

    def test_from_tangent(self):
        a = Point(-1.5, 0.5)
        b = Point(0, -1)
        c = Point(1.5, 0.5)
        d = Point(1.5, -0.5)
        l = Line(0, 1, -1)

        conic = Conic.from_tangent(l, a, b, c, d)

        assert conic.contains(a)
        assert conic.contains(b)
        assert conic.contains(c)
        assert conic.contains(d)
        assert conic.is_tangent(l)

    def test_from_crossratio(self):
        a = Point(0, 1)
        b = Point(0, -1)
        c = Point(1.5, 0.5)
        d = Point(1.5, -0.5)
        e = Point(-1.5, 0.5)

        conic1 = Conic.from_points(a, b, c, d, e)
        cr = crossratio(a, b, c, d, e)
        conic2 = Conic.from_crossratio(cr, a, b, c, d)

        assert conic1 == conic2

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

    def test_foci(self):
        f1 = Point(0, np.sqrt(5))
        f2 = Point(0, -np.sqrt(5))
        b = Point(0, 3)

        conic = Conic.from_foci(f1, f2, b)
        f = conic.foci

        assert conic.contains(b)
        assert len(f) == 2
        assert f1 in f and f2 in f


class TestCircle:

    def test_contains(self):
        c = Circle(Point(0, 1), 1)
        assert c.contains(Point(0, 2))
        assert c.contains(Point(1, 1))

    def test_center(self):
        c = Circle(Point(0, 1), 1)
        assert c.center == Point(0, 1)

    def test_intersection_angle(self):
        c1 = Circle()
        c2 = Circle(Point(1, 1), 1)
        assert np.isclose(c1.intersection_angle(c2), np.pi/2)


class TestQuadric:

    def test_tangent(self):
        q = Quadric([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, -1]])

        assert q.contains(Point(1, 0, 0))
        assert q.tangent(at=Point(1, 0, 0)) == Plane(Point(1, 0, 0), Point(1, 0, 1), Point(1, 1, 0))

    def test_components(self):
        e = Plane(1, 2, 3, 4)
        f = Plane(4, 3, 2, 1)

        q = Quadric.from_planes(e, f)
        assert q.components == [e, f]

    def test_intersection(self):
        s = Sphere(Point(0, 0, 2), 2)
        l = Line(Point(-1, 0, 2), Point(1, 0, 2))

        assert s.contains(Point(0, 0, 0))
        assert s.intersect(l) == [Point(-2, 0, 2), Point(2, 0, 2)]

        c = Cone(vertex=Point(1, 1, 1), base_center=Point(2, 2, 2), radius=2)
        a = np.sqrt(2)
        l = Line(Point(0, 4, 2), Point(4, 0, 2))
        assert c.intersect(l) == [Point(2-a, 2+a, 2), Point(2+a, 2-a, 2)]

        s = Sphere(Point(0, 0, 0, 2), 2)
        l = Line(Point(-1, 0, 0, 2), Point(1, 0, 0, 2))

        assert s.contains(Point(0, 0, 0, 0))
        assert s.intersect(l) == [Point(2, 0, 0, 2), Point(-2, 0, 0, 2)]

    def test_sphere(self):
        s2 = Sphere()
        s3 = Sphere(Point(1, 2, 3, 4), 5)

        assert s2.center == Point(0, 0, 0)
        assert np.isclose(s2.radius, 1)
        assert np.isclose(s2.volume, 4/3*np.pi)
        assert np.isclose(s2.area, 4*np.pi)

        assert s3.center == Point(1, 2, 3, 4)
        assert np.isclose(s3.radius, 5)
        assert np.isclose(s3.volume, 1/2 * np.pi**2 * 5**4)
        assert np.isclose(s3.area, 2 * np.pi**2 * 5**3)

    def test_cone(self):
        c = Cone(vertex=Point(1, 0, 0), base_center=Point(2, 0, 0), radius=4)

        assert c.contains(Point(1, 0, 0))
        assert c.contains(Point(2, 4, 0))
        assert c.contains(Point(2, 0, 4))
        assert c.contains(Point(0, 0, -4))

        c = Cone(vertex=Point(1, 1, 1), base_center=Point(2, 2, 2), radius=2)

        s = np.sqrt(2)
        assert c.contains(Point(1, 1, 1))
        assert c.contains(Point(2+s, 2-s, 2))
        assert c.contains(Point(2, 2-s, 2+s))
        assert c.contains(Point(s, 0, -s))

    def test_cylinder(self):
        c = Cylinder(center=Point(1, 0, 0), direction=Point(1, 0, 0), radius=4)

        assert c.contains(Point(1, 0, 4))
        assert c.contains(Point(2, 4, 0))
        assert c.contains(Point(2, 0, 4))
        assert c.contains(Point(-1, 0, -4))

        c = Cylinder(direction=Point(1, 1, 1), radius=2)

        s = np.sqrt(2)
        assert c.contains(Point(2 + s, 2 - s, 2))
        assert c.contains(Point(2, 2 - s, 2 + s))
        assert c.contains(Point(s, 0, -s))
        assert c.contains(Point(42+s, 42, 42-s))
