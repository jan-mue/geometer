from geometer import Point, Segment, Rectangle, Triangle
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
        assert s1.intersect(s2) == Point(1, 1)


class TestPolygon:

    def test_intersections(self):
        a = Point(0, 0)
        b = Point(0, 2)
        c = Point(2, 2)
        d = Point(2, 0)
        r = Rectangle(a, b, c, d)
        s = Segment(a, c)
        assert r.intersect(s) == [a, c]

    def test_area(self):
        a = Point(0, 0)
        b = Point(2, 0)
        c = Point(0, 2)
        t = Triangle(a,b,c)
        assert np.isclose(t.area(), 2)
