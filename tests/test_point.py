import numpy as np
from geometer import Point, Line, Plane, PointCollection, LineCollection, PlaneCollection, join, meet, is_perpendicular, translation, rotation


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

        p = Point([1, 0, 0])
        q = Point(0, 1)
        assert 2*p + 3*q == Point(2, 3)

    def test_parallel(self):
        p = Point(0, 1)
        q = Point(1, 1)
        r = Point(0, 0)
        l = Line(p, q)
        m = l.parallel(through=r)

        assert m == Line(0, 1, 0)
        assert l.is_parallel(m)

    def test_perpendicular(self):
        p = Point(1, 1)
        l = Line(1, 1, 0)
        m = l.perpendicular(p)

        assert m == Line(-1, 1, 0)

        m = l.perpendicular(Point(0, 0))
        assert m == Line(-1, 1, 0)

        p = Point(1, 1, 0)
        q = Point(0, 0, 1)
        l = Line(p, q)
        m = l.perpendicular(p)

        assert is_perpendicular(l, m)


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

        # two lines
        m = Line(Point(0, 0, 0), Point(1, 2, 0))
        assert join(l, m) == Plane(0, 0, 1, 0)

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

    def test_project(self):
        p1 = Point(1, 1, 0)
        p2 = Point(2, 1, 0)
        l = Line(p1, p2)
        assert l.project(Point(0, 0, 0)) == Point(0, 1, 0)

        e = Plane(0, 0, 1, 0)
        assert e.project(Point(1, 1, 5)) == p1

    def test_parallel(self):
        p = Point(0, 0, 1)
        q = Point(1, 0, 1)
        r = Point(0, 1, 1)
        e = Plane(p, q, r)
        f = e.parallel(through=Point(0, 0, 0))
        assert f == Plane(0, 0, 1, 0)
        assert e.is_parallel(f)

    def test_perpendicular(self):
        p = Point(1, 1, 0)
        q = Point(0, 0, 1)
        l = Line(p, q)
        m = l.perpendicular(p)

        assert is_perpendicular(l, m)

        r = Point(1, 2, 3)
        e = Plane(p, q, r)
        m = e.perpendicular(p)

        assert is_perpendicular(l, m)


class Test4D:

    def test_join(self):
        p1 = Point(1, 1, 4, 0)
        p2 = Point(2, 1, 5, 0)
        p3 = Point(3, 4, 6, 0)
        p4 = Point(0, 2, 7, 0)
        p5 = Point(1, 5, 8, 0)

        # 4 points
        assert join(p1, p2, p3, p4).contains(p5)

        # 3 points
        assert join(p1, p2, p3).contains(p3)

        # two lines
        l = Line(p1, p2)
        m = Line(p3, p4)
        assert join(l, m) == Plane(p1, p2, p3, p4)

        # coplanar lines
        l = Line(p1, p2)
        m = Line(p1, p3)
        assert join(l, m).contains(p3)

        # point and line
        p = join(l, p3)
        assert p == join(p1, p2, p3)

        # 2 points
        l = p1.join(p2)
        assert l.contains(Point(3, 1, 6, 0))

    def test_meet(self):
        p1 = Plane(1, 0, 0, 0, 0)
        p2 = Plane(0, 1, 0, 0, 0)
        p3 = Plane(0, 0, 1, 0, 0)
        p4 = Plane(0, 0, 0, 1, 0)

        # four hyperplanes
        assert meet(p1, p2, p3, p4) == Point(0, 0, 0, 0)

        # hyperplane and line
        l = Line(Point(0, 0, 0, 0), Point(0, 0, 1, 0))
        assert p3.meet(l) == Point(0, 0, 0, 0)

        # two lines
        m = Line(Point(0, 0, 0, 0), Point(1, 2, 5, 6))
        assert l.meet(m) == Point(0, 0, 0, 0)

    def test_project(self):
        p1 = Point(1, 0, 0, 0)
        p2 = Point(0, 1, 0, 0)

        l = Line(p1, p2)
        assert l.project(Point(0, 0, 0, 0)) == Point(0.5, 0.5, 0, 0)


class TestCollections:

    def test_join(self):
        a = PointCollection([Point(0, 0), Point(0, 1)])
        b = PointCollection([Point(1, 0), Point(1, 1)])

        assert a.join(b) == LineCollection([Line(0, 1, 0), Line(0, 1, -1)])

        a = PointCollection([Point(0, 0, 0), Point(0, 0, 1)])
        b = PointCollection([Point(1, 0, 0), Point(1, 0, 1)])
        c = PointCollection([Point(0, 1, 0), Point(0, 1, 1)])

        assert join(a, b, c) == PlaneCollection([Plane(0, 0, 1, 0), Plane(0, 0, 1, -1)])
        assert join(a, b.join(c)) == PlaneCollection([Plane(0, 0, 1, 0), Plane(0, 0, 1, -1)])

    def test_meet(self):
        a = LineCollection([Line(0, 1, 0), Line(0, 1, -1)])
        b = LineCollection([Line(1, 0, 0), Line(1, 0, -1)])

        assert a.meet(b) == PointCollection([Point(0, 0), Point(1, 1)])

        a = LineCollection([Line(Point(0, 0, 0), Point(0, 0, 1)), Line(Point(1, 0, 0), Point(1, 0, 1))])
        b = LineCollection([Line(Point(0, 0, 0), Point(0, 1, 0)), Line(Point(1, 0, 0), Point(1, 1, 0))])

        assert a.meet(b) == PointCollection([Point(0, 0, 0), Point(1, 0, 0)])

        a = LineCollection([Line(Point(0, 0, 0), Point(0, 0, 1)), Line(Point(1, 0, 0), Point(1, 0, 1))])
        b = PlaneCollection([Plane(0, 0, 1, 0), Plane(0, 0, 1, -1)])

        assert a.meet(b) == PointCollection([Point(0, 0, 0), Point(1, 0, 1)])

    def test_homogenize(self):
        a = PointCollection([(0, 0), (0, 1)])
        b = PointCollection([Point(0, 0), Point(0, 1)])

        assert a == b

    def test_arithmetic(self):
        a = PointCollection([Point(0, 1), Point(0, 1)])
        b = PointCollection([Point(1, 0), Point(1, 0)])
        c = PointCollection([Point(1, 1), Point(1, 1)])

        assert a + b == c
        assert a - c == -b
        assert 2*a + 2*b == 2*c
        assert (2*a + 2*b) / 2 == c
        assert a + Point(1, 0) == c

    def test_transform(self):
        a = PointCollection([(1, 0), (0, 1)])

        assert translation(1, 1) * a == PointCollection([(2, 1), (1, 2)])
        assert rotation(np.pi/2) * a == PointCollection([(0, 1), (-1, 0)])
