import numpy as np
from geometer import (
    Point,
    Line,
    Conic,
    Circle,
    Quadric,
    Plane,
    Ellipse,
    Sphere,
    Cone,
    Cylinder,
    QuadricCollection,
)
from geometer import (
    PointCollection,
    LineCollection,
    PlaneCollection,
    crossratio,
    translation,
    rotation,
)


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

        c3 = Conic.from_lines(Line(1, 0, 0), Line(0, 1, 0))
        assert Point(1, 0) in c.intersect(c3)
        assert Point(0, 1) in c.intersect(c3)
        assert Point(-1, 0) in c.intersect(c3)
        assert Point(0, -1) in c.intersect(c3)

    def test_contains(self):
        c = Conic([[1, 0, 0], [0, 1, 0], [0, 0, -1]])

        assert c.contains(Point(1, 0))

    def test_foci(self):
        e = Ellipse(Point(0, 0), 3, 2)
        f = e.foci
        f1 = Point(np.sqrt(5), 0)
        f2 = Point(np.sqrt(5), 0)

        assert len(f) == 2
        assert f1 in f and f2 in f

    def test_from_foci(self):
        f1 = Point(0, np.sqrt(5))
        f2 = Point(0, -np.sqrt(5))
        b = Point(0, 3)
        conic = Conic.from_foci(f1, f2, b)

        assert conic == Ellipse(Point(0, 0), 2, 3)


class TestCircle:
    def test_contains(self):
        c = Circle(Point(0, 1), 1)
        assert c.contains(Point(0, 2))
        assert c.contains(Point(1, 1))

    def test_center(self):
        c = Circle(Point(0, 1), 1)
        assert c.center == Point(0, 1)

    def test_intersection(self):
        c = Circle(Point(0, 2), 2)
        l = Line(Point(-1, 2), Point(1, 2))

        assert c.contains(Point(0, 0))
        assert c.intersect(l) == [Point(-2, 2), Point(2, 2)]
        assert c.intersect(l - Point(0, 2)) == [Point(0, 0)]

        l = LineCollection(
            [Line(Point(-1, 2), Point(1, 2)), Line(Point(0, 2), Point(0, 0))]
        )
        assert c.intersect(l) == [
            PointCollection([Point(-2, 2), Point(0, 0)]),
            PointCollection([Point(2, 2), Point(0, 4)]),
        ]

    def test_intersection_angle(self):
        c1 = Circle()
        c2 = Circle(Point(1, 1), 1)
        assert np.isclose(c1.intersection_angle(c2), np.pi / 2)

    def test_copy(self):
        c1 = Circle(Point(0, 1), 4.5)
        c2 = c1.copy()

        assert c1 == c2
        assert c1 is not c2
        assert c1.center == c2.center
        assert c1.radius == c2.radius

    def test_transform(self):
        c = Circle()
        t = translation(1, 1)

        assert t * c == Circle(Point(1, 1))
        assert t.apply(c).center == Point(1, 1)


class TestQuadric:
    def test_tangent(self):
        q = Quadric([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])

        assert q.contains(Point(1, 0, 0))
        assert q.tangent(at=Point(1, 0, 0)) == Plane(
            Point(1, 0, 0), Point(1, 0, 1), Point(1, 1, 0)
        )

    def test_components(self):
        e = Plane(1, 2, 3, 4)
        f = Plane(4, 3, 2, 1)

        q = Quadric.from_planes(e, f)
        assert q.components == [e, f]


class TestSphere:
    def test_intersection(self):
        s = Sphere(Point(0, 0, 2), 2)
        l = Line(Point(-1, 0, 2), Point(1, 0, 2))

        assert s.contains(Point(0, 0, 0))
        assert s.intersect(l) == [Point(-2, 0, 2), Point(2, 0, 2)]
        assert s.intersect(l - Point(0, 0, 2)) == [Point(0, 0, 0)]

        l = LineCollection(
            [
                Line(Point(-1, 0, 2), Point(1, 0, 2)),
                Line(Point(0, 0, 0), Point(0, 0, 2)),
            ]
        )
        assert s.intersect(l) == [
            PointCollection([Point(-2, 0, 2), Point(0, 0, 0)]),
            PointCollection([Point(2, 0, 2), Point(0, 0, 4)]),
        ]

        s = Sphere(Point(0, 0, 0, 2), 2)
        l = Line(Point(-1, 0, 0, 2), Point(1, 0, 0, 2))

        assert s.contains(Point(0, 0, 0, 0))
        assert s.intersect(l) == [Point(2, 0, 0, 2), Point(-2, 0, 0, 2)]

    def test_s2(self):
        s2 = Sphere()

        assert s2.center == Point(0, 0, 0)
        assert np.isclose(s2.radius, 1)
        assert np.isclose(s2.volume, 4 / 3 * np.pi)
        assert np.isclose(s2.area, 4 * np.pi)

    def test_s3(self):
        s3 = Sphere(Point(1, 2, 3, 4), 5)

        assert s3.center == Point(1, 2, 3, 4)
        assert np.isclose(s3.radius, 5)
        assert np.isclose(s3.volume, 1 / 2 * np.pi ** 2 * 5 ** 4)
        assert np.isclose(s3.area, 2 * np.pi ** 2 * 5 ** 3)

    def test_transform(self):
        s = Sphere(Point(0, 0, 2), 2)
        t = translation(1, 1, -1)

        assert t * s == Sphere(Point(1, 1, 1), 2)

    def test_add(self):
        s = Sphere()
        p = Point(0, 0, 2)

        assert s + p == Sphere(p)
        assert s - p == Sphere(-p)


class TestCone:
    def test_intersection(self):
        c = Cone(vertex=Point(1, 1, 1), base_center=Point(2, 2, 2), radius=2)
        a = np.sqrt(2)
        l = Line(Point(0, 4, 2), Point(4, 0, 2))

        assert c.intersect(l) == [Point(2 - a, 2 + a, 2), Point(2 + a, 2 - a, 2)]

    def test_init(self):
        c = Cone(vertex=Point(1, 0, 0), base_center=Point(2, 0, 0), radius=4)

        assert c.contains(Point(1, 0, 0))
        assert c.contains(Point(2, 4, 0))
        assert c.contains(Point(2, 0, 4))
        assert c.contains(Point(0, 0, -4))

        c = Cone(vertex=Point(1, 1, 1), base_center=Point(2, 2, 2), radius=2)

        s = np.sqrt(2)
        assert c.contains(Point(1, 1, 1))
        assert c.contains(Point(2 + s, 2 - s, 2))
        assert c.contains(Point(2, 2 - s, 2 + s))
        assert c.contains(Point(s, 0, -s))

    def test_transform(self):
        c = Cone(vertex=Point(1, 1, 1), base_center=Point(2, 2, 2), radius=2)
        t = translation(-1, -1, -1)

        assert t * c == Cone(
            vertex=Point(0, 0, 0), base_center=Point(1, 1, 1), radius=2
        )


class TestCylinder:
    def test_init(self):
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
        assert c.contains(Point(42 + s, 42, 42 - s))

    def test_transform(self):
        c = Cylinder(center=Point(1, 0, 0), direction=Point(1, 0, 0), radius=4)
        t = translation(1, 1, 1)
        r = rotation(np.pi / 2, axis=Point(0, 1, 0))

        assert t * c == Cylinder(
            center=Point(2, 1, 1), direction=Point(1, 0, 0), radius=4
        )
        assert r * c == Cylinder(
            center=Point(0, 0, 1), direction=Point(0, 0, 1), radius=4
        )


class TestQuadricCollection:
    def test_contains(self):
        s = QuadricCollection([Sphere(), Sphere(radius=2)])
        p = Point(1, 0, 0)

        assert np.all(s.contains(p) == [True, False])

    def test_components(self):
        l = Line(1, 2, 3)
        m = Line(4, 5, 6)
        g = Line(3, 2, 1)
        h = Line(6, 5, 4)

        q = QuadricCollection([Conic.from_lines(l, m), Conic.from_lines(g, h)])
        assert q.components == [LineCollection([m, g]), LineCollection([l, h])]

        e = Plane(1, 2, 3, 4)
        f = Plane(4, 3, 2, 1)
        g = Plane(5, 6, 7, 8)
        h = Plane(8, 7, 6, 5)

        q = QuadricCollection([Quadric.from_planes(e, f), Quadric.from_planes(g, h)])
        assert q.components == [PlaneCollection([e, g]), PlaneCollection([f, h])]

    def test_intersection(self):
        q = QuadricCollection([Circle(Point(0, 1), 1), Circle(Point(0, 2), 2)])
        l = LineCollection(
            [Line(Point(-1, 1), Point(1, 1)), Line(Point(-1, 2), Point(1, 2))]
        )

        assert q.intersect(l) == [
            PointCollection([Point(1, 1), Point(-2, 2)]),
            PointCollection([Point(-1, 1), Point(2, 2)]),
        ]

        q = QuadricCollection([Sphere(Point(0, 0, 1), 1), Sphere(Point(0, 0, 2), 2)])
        l = LineCollection(
            [
                Line(Point(-1, 0, 1), Point(1, 0, 1)),
                Line(Point(-1, 0, 2), Point(1, 0, 2)),
            ]
        )
        m = Line(Point(-1, 0, 2), Point(1, 0, 2))

        assert q.intersect(l) == [
            PointCollection([Point(-1, 0, 1), Point(-2, 0, 2)]),
            PointCollection([Point(1, 0, 1), Point(2, 0, 2)]),
        ]
        assert q.intersect(m) == [
            PointCollection([Point(0, 0, 2), Point(-2, 0, 2)]),
            PointCollection([Point(0, 0, 2), Point(2, 0, 2)]),
        ]

    def test_tangent(self):
        q = QuadricCollection([Sphere(), Sphere(radius=2)])
        p = PointCollection([Point(1, 0, 0), Point(2, 0, 0)])

        assert all(q.contains(p))
        assert q.tangent(at=p) == PlaneCollection(
            [Plane(1, 0, 0, -1), Plane(1, 0, 0, -2)]
        )
        assert all(q.is_tangent(q.tangent(at=p)))
