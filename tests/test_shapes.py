import numpy as np
from geometer import Point, Segment, Rectangle, Simplex, Triangle, Cuboid, Line, RegularPolygon, Polygon, SegmentCollection, PointCollection, dist, rotation, translation


class TestSegment:

    def test_contains(self):
        # both points finite
        p = Point(0, 0)
        q = Point(2, 1)
        s = Segment(p, q)
        assert s.contains(p)
        assert s.contains(q)
        assert s.contains(0.5 * (p + q))
        assert not s.contains(p - q)
        assert not s.contains(Point([2, 1, 0]))

        # first point at infinity
        p = Point([-2, -1, 0])
        q = Point(2, 1)
        s = Segment(p, q)
        assert s.contains(p)
        assert s.contains(q)
        assert s.contains(0.5*q)
        assert not s.contains(2*q)

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
        assert s.contains(0.5*(p+q))
        assert not s.contains(p-q)
        assert not s.contains(Point(0, 0))

    def test_equal(self):
        p = Point(0, 0)
        q = Point(2, 1)
        s = Segment(p, q)

        assert s == Segment(q, p)
        assert s == s
        assert s == Segment([(0, 0), (2, 1)], homogenize=True)
        assert s != Segment([(0, 0), (1, 2)], homogenize=True)

    def test_intersect(self):
        a = Point(0, 0)
        b = Point(0, 2)
        c = Point(2, 2)
        d = Point(2, 0)
        s1 = Segment(a, c)
        s2 = Segment(b, d)

        assert s1.intersect(s2) == [Point(1, 1)]

    def test_midpoint(self):
        p = Point(0, 0)
        q = Point(2, 2)
        s = Segment(p, q)
        assert s.midpoint == Point(1, 1)

        p = Point(0, 0, 0)
        q = Point(0, 2, 0)
        s = Segment(p, q)
        assert s.midpoint == Point(0, 1, 0)

    def test_transformation(self):
        p = Point(0, 0)
        q = Point(2, 2)
        s = Segment(p, q)

        r = rotation(np.pi/2)
        assert r*s == Segment(p, Point(-2, 2))
        assert r.apply(s)._line == r.apply(s._line)

    def test_getitem(self):
        p = Point(0, 0)
        q = Point(2, 2)
        s = Segment(p, q)

        assert s[0] == p
        assert s[1] == q


class TestPolygon:

    def test_equal(self):
        points = np.random.rand(50, 3)

        p1 = Polygon(points)
        p2 = Polygon(*[Point(p) for p in points])

        assert p1 == p2

    def test_intersect(self):
        a = Point(0, 0)
        b = Point(0, 2)
        c = Point(2, 2)
        d = Point(2, 0)
        r = Rectangle(a, b, c, d)
        s = Segment(a, c)
        assert r.intersect(s) == [a, c]

        foo = Rectangle(Point(148.06094049635456, 10.151151779987144, 60.522099063951394),
                        Point(129.78569335065157, -42.129870038015355, 60.54878245579997),
                        Point(85.91668756014471, -26.79517716452499, 60.41723371984577),
                        Point(104.19193470584759, 25.485844653477507, 60.390550327997126))
        bar = Segment(Point(-38.9592826559563, -6.703132040294841, 64.78693707404751),
                      Point(133.01711836447913, -6.633886165038485, 54.310634812542006))

        assert len(foo.intersect(bar)) == 0

    def test_edges(self):
        a = Point(0, 0)
        b = Point(0, 2)
        c = Point(2, 2)
        d = Point(2, 0)
        r = Rectangle(a, b, c, d)
        assert r.edges == [Segment(a, b), Segment(b, c), Segment(c, d), Segment(d, a)]

    def test_contains(self):
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

        assert p.contains(a)
        assert p.contains(b)
        assert p.contains(c)
        assert p.contains(d)
        assert p.contains(Point([0, 0, 1, 0]))
        assert not p.contains(Point([1, 1, 1, 0]))

    def test_transformation(self):
        a = Point(0, 0)
        b = Point(0, 1)
        c = Point(2, 1)
        d = Point(2, 0)
        r = Rectangle(a, b, c, d)
        r2 = rotation(np.pi/2)*r

        assert r.area == r2.area
        assert r2.contains(Point(-0.5, 1.5))

        l = Line(Point(0, 0, -10), Point(0, 0, 10))
        r = Rectangle(Point(-10, -10, 0), Point(10, -10, 0), Point(10, 10, 0), Point(-10, 10, 0))
        t = rotation(np.pi/6, Point(1, 0, 0))

        assert r.intersect(l) == [Point(0, 0, 0)]
        assert (t*r).intersect(l) == [Point(0, 0, 0)]

    def test_copy(self):
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

    def test_centroid(self):
        a = Point(0, 0, 1)
        b = Point(2, 0, 1)
        c = Point(2, 2, 1)
        d = Point(0, 2, 1)
        r = Rectangle(a, b, c, d)

        assert r.centroid == Point(1, 1, 1)

    def test_getitem(self):
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

    def test_contains(self):
        a = Point(0, 0)
        b = Point(0, 2)
        c = Point(2, 1)
        t = Triangle(a, b, c)

        assert t.contains(Point(1, 1))
        assert all(t.contains(PointCollection([a, b, c])))
        assert not t.contains(Point(-1, 1))
        assert not t.contains(Point([1, 1, 0]))

    def test_area(self):
        a = Point(0, 0)
        b = Point(2, 0)
        c = Point(0, 2)
        t = Triangle(a, b, c)
        assert np.isclose(t.area, 2)

    def test_centroid(self):
        a = Point(0, 0, 1)
        b = Point(2, 0, 1)
        c = Point(2, 2, 1)
        t = Triangle(a, b, c)

        s1, s2, s3 = t.edges
        l1 = s1.midpoint.join(c)
        l2 = s2.midpoint.join(a)

        assert t.centroid == (a+b+c)/3
        assert t.centroid == l1.meet(l2)


class TestRegularPolygon:

    def test_init(self):
        a = Point(0, 0, 0)
        p = RegularPolygon(a, 1, 6, axis=Point(0, 0, 1))

        d = p.edges[0].length

        assert len(p.vertices) == 6
        assert np.isclose(dist(a, p.vertices[0]), 1)
        assert all(np.isclose(s.length, d) for s in p.edges[1:])
        assert np.allclose(p.angles, np.pi/3)
        assert p.center == a

    def test_radius(self):
        p = RegularPolygon(Point(0, 0, 0), 1, 6, axis=Point(0, 0, 1))

        assert np.isclose(p.radius, 1)
        assert np.isclose(p.inradius, np.cos(np.pi/6))

    def test_transform(self):
        p = RegularPolygon(Point(0, 0, 0), 1, 6, axis=Point(0, 0, 1))
        t = translation(1, 1, 0)

        assert t*p == RegularPolygon(Point(1, 1, 0), 1, 6, axis=Point(0, 0, 1))
        assert isinstance(t*p, RegularPolygon)
        
        
class TestSimplex:

    def test_volume(self):
        a = Point(0, 0, 0)
        b = Point(1, 0, 0)
        c = Point(0, 1, 0)
        d = Point(0, 0, 1)
        s = Simplex(a, b, c, d)

        assert np.isclose(s.volume, 1/6)

        triangle = Simplex(a, b, c)
        assert np.isclose(triangle.volume, 1/2)

    def test_transform(self):
        a = Point(0, 0, 0)
        b = Point(1, 0, 0)
        c = Point(0, 1, 0)
        d = Point(0, 0, 1)
        s = Simplex(a, b, c, d)
        x = Point(1, 1, 1)
        t = translation(x)

        assert t * s == Simplex(a + x, b + x, c + x, d + x)


class TestCuboid:

    def test_intersect(self):
        a = Point(0, 0, 0)
        b = Point(1, 0, 0)
        c = Point(0, 1, 0)
        d = Point(0, 0, 1)
        cube = Cuboid(a, b, c, d)
        l = Line(Point(2, 0.5, 0.5), Point(-1, 0.5, 0.5))
        assert cube.intersect(l) == [Point(0, 0.5, 0.5), Point(1, 0.5, 0.5)]

    def test_edges(self):
        a = Point(0, 0, 0)
        b = Point(1, 0, 0)
        c = Point(0, 1, 0)
        d = Point(0, 0, 1)
        cube = Cuboid(a, b, c, d)

        assert len(cube.edges) == 12

    def test_area(self):
        a = Point(0, 0, 0)
        b = Point(1, 0, 0)
        c = Point(0, 1, 0)
        d = Point(0, 0, 1)
        cube = Cuboid(a, b, c, d)
        assert len(cube.faces) == 6
        assert len(cube.vertices) == 8
        assert cube.area == 6

    def test_transform(self):
        a = Point(0, 0, 0)
        b = Point(1, 0, 0)
        c = Point(0, 1, 0)
        d = Point(0, 0, 1)
        cube = Cuboid(a, b, c, d)
        x = Point(1, 1, 1)
        t = translation(x)

        assert t*cube == Cuboid(a+x, b+x, c+x, d+x)
        assert isinstance(t*cube, Cuboid)

    def test_add(self):
        a = Point(0, 0, 0)
        b = Point(1, 0, 0)
        c = Point(0, 1, 0)
        d = Point(0, 0, 1)
        cube = Cuboid(a, b, c, d)
        p = Point(1, 2, 3)

        assert cube + p == Cuboid(a+p, b+p, c+p, d+p)
        assert cube - p == Cuboid(a-p, b-p, c-p, d-p)

    def test_getitem(self):
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

    def test_contains(self):
        p = PointCollection([(0, 0), (1, 0)], homogenize=True)
        q = PointCollection([(2, 1), (3, 1)], homogenize=True)
        s = SegmentCollection(p, q)

        assert all(s.contains(p))
        assert all(s.contains(q))
        assert all(s.contains(0.5 * (p + q)))
        assert not any(s.contains(p - q))
        assert not any(s.contains(Point([2, 1, 0])))
