from geometer import *


class Test2D:

    def test_join(self):
        p = Point(1, 0)
        q = Point(0, 1)
        assert p.join(q) == Line(-1, -1, 1)

    def test_meet(self):
        l = Line(-1, -1, 2)
        m = Line(1, -1, 0)
        assert l.meet(m) == Point(1, 1)

    def test_add(self):
        p = Point(1, 0)
        q = Point(0, 1)
        assert p + q == Point(1, 1)

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


class Test3D:

    def test_join(self):
        p1 = Point(1, 1, 0)
        p2 = Point(2, 1, 0)
        p3 = Point(3, 4, 0)
        p4 = Point(0, 2, 0)

        # 3 points
        assert join(p1, p2, p3).contains(p4)

        # 2 points
        l = p1.join(p2)
        assert l.contains(Point(3, 1, 0))

        # point and line
        p = join(l, p3)
        assert p.contains(p4)

    def test_meet(self):
        p1 = Plane(1, 0, 0, 0)
        p2 = Plane(0, 0, 1, 0)
        p3 = Plane(0, 1, 0, 0)

        # three planes
        assert meet(p1, p2, p3) == Point(0, 0, 0)

        # two planes
        l = p1.meet(p2)
        m = Line(Point(0, 0, 0), Point(0, 1, 0))
        assert l == m

        # two lines
        m = Line(Point(0, 0, 0), Point(1, 2, 5))
        assert l.meet(m) == Point(0, 0, 0)

        # plane and line
        assert p3.meet(l) == Point(0, 0, 0)

    def test_contains(self):
        p1 = Point(1, 1, 0)
        p2 = Point(2, 1, 0)
        p3 = Point(3, 4, 0)
        p4 = Point(0, 2, 0)

        p = Plane(p1, p2, p3)
        l = Line(p1, p2)
        assert p.contains(p4)
        assert p.contains(l)
