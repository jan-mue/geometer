from geometer import Point, Segment, Rectangle, Triangle, Polytope, Cube, Line, RegularPolygon, dist
import numpy as np


class TestSegment:

    def test_contains(self):
        p = Point(0, 0)
        q = Point(2, 1)
        s = Segment(p, q)
        assert s.contains(p)
        assert s.contains(q)
        assert s.contains(0.5*(p+q))
        assert not s.contains(p-q)

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
        assert s.midpoint() == Point(1, 1)

        p = Point(0, 0, 0)
        q = Point(0, 2, 0)
        s = Segment(p, q)
        assert s.midpoint() == Point(0, 1, 0)


class TestPolygon:

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

    def test_area(self):
        a = Point(0, 0)
        b = Point(2, 0)
        c = Point(0, 2)
        t = Triangle(a, b, c)
        assert np.isclose(t.area(), 2)

    def test_triangulation(self):
        a = Point(0, 0)
        b = Point(0, 2)
        c = Point(2, 2)
        d = Point(2, 0)
        r = Rectangle(a, b, c, d)
        t = r.triangulate()
        assert t == [Triangle(b, d, c), Triangle(d, b, a)]

    def test_regular_polygon(self):
        a = Point(0, 0, 0)
        p = RegularPolygon(a, 1, 6, axis=Point(0, 0, 1))

        d = p.sides[0].length()

        assert len(p.vertices) == 6
        assert np.isclose(dist(a, p.vertices[0]), 1)
        assert all(np.isclose(s.length(), d) for s in p.sides[1:])
        
        
class TestPolytope:

    def test_cube(self):
        a = Point(0, 0, 0)
        b = Point(1, 0, 0)
        c = Point(0, 1, 0)
        d = Point(0, 0, 1)
        cube = Cube(a, b, c, d)
        assert len(cube.sides) == 6
        assert len(cube.vertices) == 8
        assert cube.area() == 6
        assert np.isclose(cube.volume(), 1)

    def test_intersect(self):
        a = Point(0, 0, 0)
        b = Point(1, 0, 0)
        c = Point(0, 1, 0)
        d = Point(0, 0, 1)
        cube = Cube(a, b, c, d)
        l = Line(Point(2, 0.5, 0.5), Point(-1, 0.5, 0.5))
        assert cube.intersect(l) == [Point(0, 0.5, 0.5), Point(1, 0.5, 0.5)]
    
    def test_convex_hull(self):
        pts = [
            Point(0, 0, 0),
            Point(0, 1, 0),
            Point(1, 1, 0),
            Point(1, 0, 0),
            Point(0, 0, 1),
            Point(0, 1, 1),
            Point(1, 1, 1),
            Point(1, 0, 1)
        ]

        p = Polytope(*pts)
        assert p.is_convex()
