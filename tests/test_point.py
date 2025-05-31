import numpy as np

from geometer import (
    I,
    J,
    Line,
    LineCollection,
    Plane,
    PlaneCollection,
    Point,
    PointCollection,
    is_perpendicular,
    join,
    meet,
    rotation,
    translation,
)


class Test2D:
    def test_eq(self) -> None:
        p = Point(100, 0)
        q = Point(101, 0)
        assert p == p
        assert p != q

        assert 1e6 * p != 1e6 * q

        assert p == Point([100, 0, 1])
        assert p == Point([100, 0], homogenize=True)
        assert p != [100, 0]
        assert p == [100, 0, 1]
        assert p != np.array([100, 0])
        assert p == np.array([100, 0, 1])
        assert p != 100

        assert Point(1, 1) != 1
        assert Point(1) != Point(1, 1)
        assert Point(0, 0) != 0
        assert Point(0) != 0

    def test_join(self) -> None:
        p = Point(1, 0)
        q = Point(0, 1)
        assert p.join(q) == Line(-1, -1, 1)

    def test_meet(self) -> None:
        l = Line(-1, -1, 2)
        m = Line(1, -1, 0)
        assert l.meet(m) == Point(1, 1)

    def test_add(self) -> None:
        p = Point(1, 0)
        q = Point(0, 1)
        assert p + q == Point(1, 1)

        p = Point([1, 0, 0])
        q = Point(0, 1)
        assert 2 * p + 3 * q == Point(2, 3)

    def test_parallel(self) -> None:
        p = Point(0, 1)
        q = Point(1, 1)
        r = Point(0, 0)
        l = Line(p, q)
        m = l.parallel(through=r)

        assert m == Line(0, 1, 0)
        assert l.is_parallel(m)

    def test_perpendicular(self) -> None:
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

    def test_isinf(self) -> None:
        assert not Point(1, -2).isinf
        assert not Point(0, 0).isinf
        assert Point([1, -2, 0]).isinf
        assert Point([1j, -2, 0]).isinf
        assert I.isinf
        assert J.isinf

    def test_isreal(self) -> None:
        assert Point(1, -2).isreal
        assert Point(0, 0).isreal
        assert Point([1, -2, 0]).isreal
        assert not Point([1j, -2, 0]).isreal
        assert not I.isreal
        assert not J.isreal


class Test3D:
    def test_eq(self) -> None:
        p = Point(100, 0, 0)
        q = Point(101, 0, 0)
        assert p == p
        assert p != q

        assert 1e6 * p != 1e6 * q

        assert p == Point([100, 0, 0, 1])
        assert p == Point([100, 0, 0], homogenize=True)
        assert p != [100, 0, 0]
        assert p == [100, 0, 0, 1]
        assert p != np.array([100, 0, 0])
        assert p == np.array([100, 0, 0, 1])
        assert p != 100

        assert Point(1, 1, 1) != 1
        assert Point(1, 1) != Point(1, 1, 0)
        assert Point(0, 0, 0) != 0
        assert Point(0, 0) != Point(0, 0, 0)
        assert Point(1, 2) != []

    def test_join(self) -> None:
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

    def test_meet(self) -> None:
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

    def test_contains(self) -> None:
        p1 = Point(1, 1, 0)
        p2 = Point(2, 1, 0)
        p3 = Point(3, 4, 0)
        p4 = Point(0, 2, 0)

        p = Plane(p1, p2, p3)
        l = Line(p1, p2)
        assert p.contains(p4)
        assert p.contains(l)

    def test_is_coplanar(self) -> None:
        l = Line(Point(1, 1, 0), Point(2, 1, 0))
        m = Line(Point(0, 0, 0), Point(1, 2, 0))

        assert l.is_coplanar(m)

    def test_project(self) -> None:
        p1 = Point(1, 1, 0)
        p2 = Point(2, 1, 0)
        l = Line(p1, p2)
        assert l.project(Point(0, 0, 0)) == Point(0, 1, 0)

        e = Plane(0, 0, 1, 0)
        assert e.project(Point(1, 1, 5)) == p1

    def test_parallel(self) -> None:
        p = Point(0, 0, 1)
        q = Point(1, 0, 1)
        r = Point(0, 1, 1)
        e = Plane(p, q, r)
        f = e.parallel(through=Point(0, 0, 0))
        assert f == Plane(0, 0, 1, 0)
        assert e.is_parallel(f)

    def test_perpendicular(self) -> None:
        p = Point(1, 1, 0)
        q = Point(0, 0, 1)
        r = Point(1, 2, 3)
        l = Line(p, q)
        m = l.perpendicular(p)

        assert l.meet(m) == p
        assert is_perpendicular(l, m)

        m = l.perpendicular(r)

        assert is_perpendicular(l, m)

        e = Plane(l, r)
        m = e.perpendicular(p)

        assert e.meet(m) == p
        assert is_perpendicular(l, m)

        m = e.perpendicular(p + m.direction)

        assert e.meet(m) == p
        assert is_perpendicular(l, m)

        f = e.perpendicular(l)

        assert e.meet(f) == l
        assert is_perpendicular(e, f)


class TestCollections:
    def test_join(self) -> None:
        # 2 points
        a = PointCollection([Point(0, 0), Point(0, 1)])
        b = PointCollection([Point(1, 0), Point(1, 1)])

        assert a.join(b) == LineCollection([Line(0, 1, 0), Line(0, 1, -1)])

        # 3 points
        a = PointCollection([Point(0, 0, 0), Point(0, 0, 1)])
        b = PointCollection([Point(1, 0, 0), Point(1, 0, 1)])
        c = PointCollection([Point(0, 1, 0), Point(0, 1, 1)])

        assert join(a, b, c) == PlaneCollection([Plane(0, 0, 1, 0), Plane(0, 0, 1, -1)])

        # two lines
        l = a.join(b)
        m = a.join(c)
        assert join(l, m) == PlaneCollection([Plane(0, 0, 1, 0), Plane(0, 0, 1, -1)])

        # point and line
        assert join(a, b.join(c)) == PlaneCollection([Plane(0, 0, 1, 0), Plane(0, 0, 1, -1)])

    def test_meet(self) -> None:
        # three planes
        a = PlaneCollection([Plane(1, 0, 0, 0), Plane(1, 0, 0, -1)])
        b = PlaneCollection([Plane(0, 1, 0, 0), Plane(0, 1, 0, -1)])
        c = PlaneCollection([Plane(0, 0, 1, 0), Plane(0, 0, 1, -1)])
        assert meet(a, b, c) == PointCollection([Point(0, 0, 0), Point(1, 1, 1)])

        # two planes
        l = a.meet(b)
        m = LineCollection([Line(Point(0, 0, 0), Point(0, 0, 1)), Line(Point(1, 1, 0), Point(1, 1, 1))])
        assert l == m

        # two lines in 2D
        a = LineCollection([Line(0, 1, 0), Line(0, 1, -1)])  # type: ignore[assignment]
        b = LineCollection([Line(1, 0, 0), Line(1, 0, -1)])  # type: ignore[assignment]
        assert a.meet(b) == PointCollection([Point(0, 0), Point(1, 1)])

        # two lines in 3D
        a = LineCollection([Line(Point(0, 0, 0), Point(0, 0, 1)), Line(Point(1, 0, 0), Point(1, 0, 1))])  # type: ignore[assignment]
        b = LineCollection([Line(Point(0, 0, 0), Point(0, 1, 0)), Line(Point(1, 0, 0), Point(1, 1, 0))])  # type: ignore[assignment]
        assert a.meet(b) == PointCollection([Point(0, 0, 0), Point(1, 0, 0)])

        # plane and line
        a = LineCollection([Line(Point(0, 0, 0), Point(0, 0, 1)), Line(Point(1, 0, 0), Point(1, 0, 1))])  # type: ignore[assignment]
        b = PlaneCollection([Plane(0, 0, 1, 0), Plane(0, 0, 1, -1)])
        assert a.meet(b) == PointCollection([Point(0, 0, 0), Point(1, 0, 1)])

    def test_homogenize(self) -> None:
        a = PointCollection([(0, 0), (0, 1)], homogenize=True)
        b = PointCollection([Point(0, 0), Point(0, 1)])

        assert a == b

    def test_arithmetic(self) -> None:
        a = PointCollection([Point(0, 1), Point(0, 1)])
        b = PointCollection([Point(1, 0), Point(1, 0)])
        c = PointCollection([Point(1, 1), Point(1, 1)])

        assert a + b == c
        assert a - c == -b
        assert 2 * a + 2 * b == 2 * c
        assert (2 * a + 2 * b) / 2 == c
        assert a + Point(1, 0) == c

    def test_transform(self) -> None:
        a = PointCollection([(1, 0), (0, 1)], homogenize=True)

        assert translation(1, 1) * a == PointCollection([(2, 1), (1, 2)], homogenize=True)
        assert rotation(np.pi / 2) * a == PointCollection([(0, 1), (-1, 0)], homogenize=True)

    def test_basis_matrix(self) -> None:
        a = PlaneCollection([Plane(1, 0, 0, 0), Plane(0, 1, 0, 0), Plane(0, 0, 1, 0)])

        assert a.basis_matrix.shape == (3, 3, 4)
        assert np.allclose(np.matmul(a.basis_matrix, a.array[..., None]), 0)

    def test_project(self) -> None:
        p1 = PointCollection([(1, 1, 0), (1, 1, 5)], homogenize=True)
        p2 = PointCollection([(2, 1, 0), (2, 1, 5)], homogenize=True)
        p3 = PointCollection([(0, 0, 0), (0, 0, 5)], homogenize=True)
        l = LineCollection(p1, p2)
        assert l.project(p3) == PointCollection([(0, 1, 0), (0, 1, 5)], homogenize=True)

        e = PlaneCollection([(0, 1, 0, -1), (0, 1, 0, -2)])
        assert e.project(p3) == PointCollection([(0, 1, 0), (0, 2, 5)], homogenize=True)

    def test_perpendicular(self) -> None:
        p1 = PointCollection([(1, 1, 0), (1, 1, 5)], homogenize=True)
        p2 = PointCollection([(2, 1, 0), (2, 1, 5)], homogenize=True)
        p3 = PointCollection([(0, 0, 0), (0, 0, 5)], homogenize=True)
        l = LineCollection(p1, p2)
        m = l.perpendicular(p1)

        assert l.meet(m) == p1
        assert all(is_perpendicular(l, m))

        m = l.perpendicular(p3 + PointCollection([(1, 1, 0), (0, 0, 0)], homogenize=True))

        assert all(is_perpendicular(l, m))

        e = PlaneCollection(l, p3)
        m = e.perpendicular(p1)

        assert e.meet(m) == p1
        assert all(is_perpendicular(l, m))

        m = e.perpendicular(p1 + PointCollection([m.direction[0], Point(0, 0, 0)]))

        assert e.meet(m) == p1
        assert all(is_perpendicular(l, m))

        f = e.perpendicular(l)

        assert e.meet(f) == l
        assert all(is_perpendicular(e, f))

    def test_isinf(self) -> None:
        assert np.all(~PointCollection([(1, -2), (0, 0)], homogenize=True).isinf)
        assert np.all(PointCollection([[1, -2, 0], [1j, -2, 0]]).isinf)  # type: ignore[arg-type]
        assert np.all(PointCollection([I, J]).isinf)

    def test_isreal(self) -> None:
        assert np.all(PointCollection([(1, -2, 1), (0, 0, 1), (1, -2, 0)]).isreal)
        assert np.all(~PointCollection([Point([1j, -2, 0]), I, J]).isreal)
