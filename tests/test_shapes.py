import numpy as np

from geometer import (
    Cuboid,
    Line,
    LineCollection,
    Point,
    PointCollection,
    Polygon,
    PolygonCollection,
    Rectangle,
    RegularPolygon,
    Segment,
    SegmentCollection,
    Simplex,
    TransformationCollection,
    Triangle,
    dist,
    rotation,
    translation,
)


class TestSegment:
    def test_contains(self) -> None:
        # both points finite
        p = Point(0, 0)
        q = Point(2, 1)
        s = Segment(p, q)
        assert s.contains(p)
        assert s.contains(q)
        assert s.contains(0.5 * (p + q))  # type: ignore[arg-type]
        assert not s.contains(p - q)
        assert not s.contains(Point([2, 1, 0]))

        # first point at infinity
        p = Point([-2, -1, 0])
        q = Point(2, 1)
        s = Segment(p, q)
        assert s.contains(p)
        assert s.contains(q)
        assert s.contains(0.5 * q)  # type: ignore[arg-type]
        assert not s.contains(2 * q)  # type: ignore[arg-type]

        # second point at infinity
        p = Point(0, 0)
        q = Point([2, 1, 0])
        s = Segment(p, q)
        assert s.contains(p)
        assert s.contains(q)
        assert s.contains(Point(2, 1))
        assert not s.contains(Point(-2, -1))

        # both points at infinity
        p = Point([-2, 1, 0])
        q = Point([2, 1, 0])
        s = Segment(p, q)
        assert s.contains(p)
        assert s.contains(q)
        assert s.contains(0.5 * (p + q))  # type: ignore[arg-type]
        assert not s.contains(p - q)
        assert not s.contains(Point(0, 0))

    def test_equal(self) -> None:
        p = Point(0, 0)
        q = Point(2, 1)
        s = Segment(p, q)

        assert s == Segment(q, p)
        assert s == s
        assert s == Segment([(0, 0), (2, 1)], homogenize=True)  # type: ignore[call-arg]
        assert s != Segment([(0, 0), (1, 2)], homogenize=True)  # type: ignore[call-arg]

    def test_intersect(self) -> None:
        a = Point(0, 0)
        b = Point(0, 2)
        c = Point(2, 2)
        d = Point(2, 0)
        s1 = Segment([a, c])
        s2 = Segment([b, d])

        assert s1.intersect(s2) == [Point(1, 1)]

    def test_midpoint(self) -> None:
        p = Point(0, 0)
        q = Point(2, 2)
        s = Segment(p, q)
        assert s.midpoint == Point(1, 1)

        p = Point(0, 0, 0)
        q = Point(0, 2, 0)
        s = Segment(p, q)
        assert s.midpoint == Point(0, 1, 0)

    def test_transformation(self) -> None:
        p = Point(0, 0)
        q = Point(2, 2)
        s = Segment(p, q)

        r = rotation(np.pi / 2)
        rotated_line = r.apply(s._line)
        assert r * s == Segment(p, Point(-2, 2))
        assert r.apply(s)._line == rotated_line

        t = TransformationCollection([r] * 3).expand_dims(1)
        result = t * s

        assert isinstance(result, SegmentCollection)
        assert result == SegmentCollection([Segment(p, Point(-2, 2))] * 3)
        assert result._line == LineCollection([rotated_line, rotated_line, rotated_line])

    def test_getitem(self) -> None:
        p = Point(0, 0)
        q = Point(2, 2)
        s = Segment(p, q)

        assert s[0] == p
        assert s[1] == q


class TestPolygon:
    def test_equal(self, rng: np.random.Generator) -> None:
        points = rng.random((50, 3))

        p1 = Polygon(points)
        p2 = Polygon(*[Point(p) for p in points])
        p3 = Polygon(np.roll(points, 42, axis=0))
        p4 = Polygon(np.flip(points, axis=0))

        assert p1 == p2
        assert p1 == p3
        assert p1 == p4

    def test_intersect(self) -> None:
        a = Point(0, 0)
        b = Point(0, 2)
        c = Point(2, 2)
        d = Point(2, 0)
        r = Rectangle(a, b, c, d)
        s = Segment(a, c)
        assert r.intersect(s) == [a, c]

        foo = Rectangle(
            Point(148.06094049635456, 10.151151779987144, 60.522099063951394),
            Point(129.78569335065157, -42.129870038015355, 60.54878245579997),
            Point(85.91668756014471, -26.79517716452499, 60.41723371984577),
            Point(104.19193470584759, 25.485844653477507, 60.390550327997126),
        )
        bar = Segment(
            Point(-38.9592826559563, -6.703132040294841, 64.78693707404751),
            Point(133.01711836447913, -6.633886165038485, 54.310634812542006),
        )

        assert len(foo.intersect(bar)) == 0

        p1 = Point(0, 0)
        p2 = Point(100, 0)
        p3 = Point(100, 100)
        p4 = Point(0, 100)
        square = Polygon(p1, p2, p3, p4)
        line1 = Line(p3, p4)
        line2 = Line(Point(100, 100 - 1e-8), Point(0, 100 + 1e-8))

        assert square.intersect(line1) == [p3, p4]
        assert len(square.intersect(line2)) == 2

    def test_edges(self) -> None:
        a = Point(0, 0)
        b = Point(0, 2)
        c = Point(2, 2)
        d = Point(2, 0)
        r = Rectangle(a, b, c, d)
        assert r.edges == [Segment(a, b), Segment(b, c), Segment(c, d), Segment(d, a)]

    def test_contains(self) -> None:
        a = Point(0, 0)
        b = Point(0, 2)
        c = Point(2, 2)
        d = Point(2, 0)
        r = Rectangle(a, b, c, d)

        assert r.contains(a)
        assert r.contains(b)
        assert r.contains(c)
        assert r.contains(d)
        assert r.contains(Point(1, 1))
        assert not r.contains(Point([1, 1, 0]))

        a = Point(0, 0, 1)
        b = Point(1, 3, 1)
        c = Point(2, 0, 1)
        d = Point(1, 1, 1)
        p = Polygon(a, b, c, d)

        assert p.contains(Point(0.5, 1, 1))
        assert not p.contains(Point(0.5, 1, 0))
        assert np.all(p.contains(PointCollection([Point(0.5, 1, 1), Point(1.5, 1, 1)])))
        assert np.all(p.contains(PointCollection([a, c, d])))

        a = Point([1, 1, 2, 0])
        b = Point([-1, 1, 2, 0])
        c = Point([-1, -1, 2, 0])
        d = Point([1, -1, 2, 0])
        p = Polygon(a, b, c, d)

        assert all(p.contains(PointCollection([a, b, c, d])))
        assert p.contains(Point([0, 0, 1, 0]))
        assert not p.contains(Point([1, 1, 1, 0]))

    def test_transformation(self) -> None:
        a = Point(0, 0)
        b = Point(0, 1)
        c = Point(2, 1)
        d = Point(2, 0)
        r = Rectangle(a, b, c, d)
        r2 = rotation(np.pi / 2) * r

        assert r.area == r2.area
        assert r2.contains(Point(-0.5, 1.5))

        l = Line(Point(0, 0, -10), Point(0, 0, 10))
        r = Rectangle(Point(-10, -10, 0), Point(10, -10, 0), Point(10, 10, 0), Point(-10, 10, 0))
        t = rotation(np.pi / 6, Point(1, 0, 0))

        assert r.intersect(l) == [Point(0, 0, 0)]
        assert (t * r).intersect(l) == [Point(0, 0, 0)]

    def test_copy(self) -> None:
        a = Point(0, 0)
        b = Point(0, 2)
        c = Point(2, 2)
        d = Point(2, 0)

        r1 = Rectangle(a, b, c, d)
        p1 = RegularPolygon(a, 1, 6)

        r2 = r1.copy()
        p2 = p1.copy()

        assert r1 == r2
        assert r1 is not r2
        assert r1.vertices == r2.vertices
        assert p1 == p2
        assert p1 is not p2
        assert p1.vertices == p2.vertices

    def test_centroid(self) -> None:
        a = Point(0, 0, 1)
        b = Point(2, 0, 1)
        c = Point(2, 2, 1)
        d = Point(0, 2, 1)
        r = Rectangle(a, b, c, d)

        assert r.centroid == Point(1, 1, 1)

    def test_getitem(self) -> None:
        a = Point(0, 0, 1)
        b = Point(2, 0, 1)
        c = Point(2, 2, 1)
        d = Point(0, 2, 1)
        r = Rectangle(a, b, c, d)

        assert r[0] == a
        assert r[1] == b
        assert r[2] == c
        assert r[3] == d


class TestTriangle:
    def test_contains(self) -> None:
        a = Point(0, 0)
        b = Point(0, 2)
        c = Point(2, 1)
        t = Triangle(a, b, c)

        assert t.contains(Point(1, 1))
        assert all(t.contains(PointCollection([a, b, c])))
        assert not t.contains(Point(-1, 1))
        assert not t.contains(Point([1, 1, 0]))

    def test_area(self) -> None:
        a = Point(0, 0)
        b = Point(2, 0)
        c = Point(0, 2)
        t = Triangle(a, b, c)
        assert np.isclose(t.area, 2)

    def test_centroid(self) -> None:
        a = Point(0, 0, 1)
        b = Point(2, 0, 1)
        c = Point(2, 2, 1)
        t = Triangle(a, b, c)

        s1, s2, s3 = t.edges
        l1 = s1.midpoint.join(c)
        l2 = s2.midpoint.join(a)

        assert t.centroid == (a + b + c) / 3
        assert t.centroid == l1.meet(l2)

    def test_circumcenter(self) -> None:
        a = Point(0, 0, 1)
        b = Point(2, 0, 1)
        c = Point(2, 2, 1)
        t = Triangle(a, b, c)

        assert t.circumcenter == Point(1, 1, 1)


class TestRegularPolygon:
    def test_init(self) -> None:
        a = Point(0, 0, 0)
        p = RegularPolygon(a, 1, 6, axis=Point(0, 0, 1))

        d = p.edges[0].length  # type: ignore[union-attr]

        assert len(p.vertices) == 6
        assert np.isclose(dist(a, p.vertices[0]), 1)
        assert all(np.isclose(p.edges[1:].length, d))  # type: ignore[union-attr, arg-type]
        assert np.allclose(p.angles, np.pi / 3)
        assert p.center == a

    def test_radius(self) -> None:
        p = RegularPolygon(Point(0, 0, 0), 1, 6, axis=Point(0, 0, 1))

        assert np.isclose(p.radius, 1)
        assert np.isclose(p.inradius, np.cos(np.pi / 6))

    def test_transform(self) -> None:
        p = RegularPolygon(Point(0, 0, 0), 1, 6, axis=Point(0, 0, 1))
        t = translation(1, 1, 0)

        assert t * p == RegularPolygon(Point(1, 1, 0), 1, 6, axis=Point(0, 0, 1))
        assert isinstance(t * p, RegularPolygon)


class TestSimplex:
    def test_volume(self) -> None:
        a = Point(0, 0, 0)
        b = Point(1, 0, 0)
        c = Point(0, 1, 0)
        d = Point(0, 0, 1)
        s = Simplex(a, b, c, d)

        assert np.isclose(s.volume, 1 / 6)

        triangle = Simplex(a, b, c)
        assert np.isclose(triangle.volume, 1 / 2)

    def test_transform(self) -> None:
        a = Point(0, 0, 0)
        b = Point(1, 0, 0)
        c = Point(0, 1, 0)
        d = Point(0, 0, 1)
        s = Simplex(a, b, c, d)
        x = Point(1, 1, 1)
        t = translation(x)

        assert t * s == Simplex(a + x, b + x, c + x, d + x)  # type: ignore[arg-type]


class TestCuboid:
    def test_intersect(self) -> None:
        a = Point(0, 0, 0)
        b = Point(1, 0, 0)
        c = Point(0, 1, 0)
        d = Point(0, 0, 1)
        cube = Cuboid(a, b, c, d)
        l = Line(Point(2, 0.5, 0.5), Point(-1, 0.5, 0.5))
        assert cube.intersect(l) == [Point(0, 0.5, 0.5), Point(1, 0.5, 0.5)]

        l = Line(a, Point(-1, 0, 0))
        assert cube.intersect(l) == [Point(0, 0, 0), Point(1, 0, 0)]

    def test_edges(self) -> None:
        a = Point(0, 0, 0)
        b = Point(1, 0, 0)
        c = Point(0, 1, 0)
        d = Point(0, 0, 1)
        cube = Cuboid(a, b, c, d)

        assert len(cube.edges) == 12

    def test_area(self) -> None:
        a = Point(0, 0, 0)
        b = Point(1, 0, 0)
        c = Point(0, 1, 0)
        d = Point(0, 0, 1)
        cube = Cuboid(a, b, c, d)
        assert len(cube.faces) == 6
        assert len(cube.vertices) == 8
        assert cube.area == 6

    def test_transform(self) -> None:
        a = Point(0, 0, 0)
        b = Point(1, 0, 0)
        c = Point(0, 1, 0)
        d = Point(0, 0, 1)
        cube = Cuboid(a, b, c, d)
        x = Point(1, 1, 1)
        t = translation(x)

        assert t * cube == Cuboid(a + x, b + x, c + x, d + x)  # type: ignore[arg-type]
        assert isinstance(t * cube, Cuboid)

    def test_add(self) -> None:
        a = Point(0, 0, 0)
        b = Point(1, 0, 0)
        c = Point(0, 1, 0)
        d = Point(0, 0, 1)
        cube = Cuboid(a, b, c, d)
        p = Point(1, 2, 3)

        assert cube + p == Cuboid(a + p, b + p, c + p, d + p)  # type: ignore[arg-type]
        assert cube - p == Cuboid(a - p, b - p, c - p, d - p)  # type: ignore[arg-type]

    def test_getitem(self) -> None:
        a = Point(0, 0, 0)
        b = Point(1, 0, 0)
        c = Point(0, 1, 0)
        d = Point(0, 0, 1)
        cube = Cuboid(a, b, c, d)

        x, y, z = b - a, c - a, d - a
        yz = Rectangle(a, a + z, a + y + z, a + y)
        xz = Rectangle(a, a + x, a + x + z, a + z)
        xy = Rectangle(a, a + x, a + x + y, a + y)

        assert isinstance(cube[0], Polygon)
        assert cube[0] == yz
        assert cube[1] == xz
        assert cube[2] == xy
        assert cube[0, 0] == a


class TestSegmentCollection:
    def test_contains(self) -> None:
        p = PointCollection([(0, 0), (1, 0)], homogenize=True)
        q = PointCollection([(2, 1), (3, 1)], homogenize=True)
        s = SegmentCollection(p, q)

        assert all(s.contains(p))
        assert all(s.contains(q))
        assert all(s.contains(0.5 * (p + q)))  # type: ignore[arg-type]
        assert not any(s.contains(p - q))
        assert not any(s.contains(Point([2, 1, 0])))

    def test_midpoint(self) -> None:
        p = PointCollection([(0, 0), (2, 1)], homogenize=True)
        q = PointCollection([(2, 2), (4, 1)], homogenize=True)
        s = SegmentCollection(p, q)
        assert s.midpoint == PointCollection([(1, 1), (3, 1)], homogenize=True)

        p = PointCollection([(0, 0, 0), (1, 1, 1)], homogenize=True)
        q = PointCollection([(0, 2, 0), (3, 3, 3)], homogenize=True)
        s = SegmentCollection(p, q)
        assert s.midpoint == PointCollection([(0, 1, 0), (2, 2, 2)], homogenize=True)

    def test_intersect(self) -> None:
        a = PointCollection([(0, 0), (0, 0)], homogenize=True)
        b = PointCollection([(2, 0), (4, 0)], homogenize=True)
        c = PointCollection([(2, 2), (4, 4)], homogenize=True)
        d = PointCollection([(0, 2), (0, 4)], homogenize=True)
        s1 = SegmentCollection(a, c)
        s2 = SegmentCollection(b, d)

        assert s1.intersect(s2) == PointCollection([(1, 1), (2, 2)], homogenize=True)
        assert s1.intersect(Line(Point(0, 0), Point(1, 1))) == []


class TestPolygonCollection:
    def test_equal(self, rng: np.random.Generator) -> None:
        points = rng.random((60, 50, 3))

        p1 = PolygonCollection(points)
        p2 = PolygonCollection([Polygon(*[Point(p) for p in poly]) for poly in points])

        assert p1 == p2
