from geometer import Point, Segment, Rectangle, Simplex, Triangle, Cuboid, Line, RegularPolygon, Polygon, dist, rotation
import numpy as np


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

        # both points  at infinity
        p = Point([-2, 1, 0])
        q = Point([2, 1, 0])
        s = Segment(p, q)
        assert s.contains(p)
        assert s.contains(q)
        assert s.contains(0.5*(p+q))
        assert not s.contains(p-q)
        assert not s.contains(Point(0, 0))

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

    def test_contains(self):
        a = Point(0, 0)
        b = Point(0, 2)
        c = Point(2, 2)
        d = Point(2, 0)
        r = Rectangle(a, b, c, d)
        assert r.contains(Point(1, 1))

        a = Point(0, 0, 1)
        b = Point(1, 3, 1)
        c = Point(2, 0, 1)
        d = Point(1, 1, 1)
        p = Polygon(a, b, c, d)

        assert p.contains(Point(0.5, 1, 1))
        assert not p.contains(Point(0.5, 1, 0))

    def test_area(self):
        a = Point(0, 0)
        b = Point(2, 0)
        c = Point(0, 2)
        t = Triangle(a, b, c)
        assert np.isclose(t.area, 2)

    def test_transformation(self):
        a = Point(0, 0)
        b = Point(0, 1)
        c = Point(2, 1)
        d = Point(2, 0)
        r = Rectangle(a, b, c, d)
        r2 = rotation(np.pi/2)*r

        assert r.area == r2.area
        assert r2.contains(Point(-0.5, 1.5))

    def test_regular_polygon(self):
        a = Point(0, 0, 0)
        p = RegularPolygon(a, 1, 6, axis=Point(0, 0, 1))

        d = p.edges[0].length

        assert len(p.vertices) == 6
        assert np.isclose(dist(a, p.vertices[0]), 1)
        assert all(np.isclose(s.length, d) for s in p.edges[1:])
        assert np.allclose(p.angles, np.pi/3)
        
        
class TestPolytope:

    def test_cube(self):
        a = Point(0, 0, 0)
        b = Point(1, 0, 0)
        c = Point(0, 1, 0)
        d = Point(0, 0, 1)
        cube = Cuboid(a, b, c, d)
        assert len(cube.faces) == 6
        assert len(cube.vertices) == 8
        assert cube.area == 6

    def test_intersect(self):
        a = Point(0, 0, 0)
        b = Point(1, 0, 0)
        c = Point(0, 1, 0)
        d = Point(0, 0, 1)
        cube = Cuboid(a, b, c, d)
        l = Line(Point(2, 0.5, 0.5), Point(-1, 0.5, 0.5))
        assert cube.intersect(l) == [Point(0, 0.5, 0.5), Point(1, 0.5, 0.5)]

    def test_simplex_volume(self):
        a = Point(0, 0, 0)
        b = Point(1, 0, 0)
        c = Point(0, 1, 0)
        d = Point(0, 0, 1)
        s = Simplex(a, b, c, d)

        assert np.isclose(s.volume, 1/6)

        triangle = Simplex(a, b, c)
        assert np.isclose(triangle.volume, 1/2)
