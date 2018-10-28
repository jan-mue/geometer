from geometer import *


class TestPoint:

    def test_add(self):
        p = Point(1, 0)
        q = Point(0, 1)
        assert p + q == Point(1, 1)


class TestLine:

    def test_parallel(self):
        p = Point(0,1)
        q = Point(1,1)
        r = Point(0,0)
        l = Line(p, q)
        m = l.parallel(through=r)
        assert m == Line(0,1,0)

    def test_perpendicular(self):
        p = Point(1,1)
        l = Line(1,1,0)
        m = l.perpendicular(p)
        assert m == Line(-1,1,0)

    def test_3d(self):
        p1 = Point(1,1,0)
        p2 = Point(2,1,0)
        p3 = Point(3,4,0)
        p4 = Point(0,2,0)
        assert join(p1,p2,p3).contains(p4)
