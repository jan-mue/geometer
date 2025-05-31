import numpy as np

from geometer import (
    Cuboid,
    Line,
    LineCollection,
    Plane,
    PlaneCollection,
    Point,
    PointCollection,
    Rectangle,
    Segment,
    Transformation,
    angle,
    angle_bisectors,
    crossratio,
    dist,
    harmonic_set,
    is_cocircular,
    is_collinear,
    is_coplanar,
    is_perpendicular,
    rotation,
)


def test_is_collinear() -> None:
    p1 = Point(1, 0)
    p2 = Point(2, 0)
    p3 = Point(3, 0)
    l = Line(p1, p2)
    assert l.contains(p3)
    assert is_collinear(p1, p2, p3)

    p1 = PointCollection([(1, 0), (1, 1)], homogenize=True)  # type: ignore[assignment]
    p2 = PointCollection([(2, 0), (2, 1)], homogenize=True)  # type: ignore[assignment]
    p3 = PointCollection([(3, 0), (3, 1)], homogenize=True)  # type: ignore[assignment]
    p4 = PointCollection([(4, 0), (4, 1)], homogenize=True)

    assert all(is_collinear(p1, p2, p3, p4))


def test_dist() -> None:
    p = Point(0, 0)
    q = Point(1, 0)
    assert np.isclose(dist(p, q), 1)

    p = Point(1000000, 0)
    q = Point(1000001, 0)
    assert np.isclose(dist(p, q), 1)

    p1 = Point(1j, 0, 2j)
    p2 = Point(0, 2j, 0)
    assert np.isclose(dist(p1, p2), 3)

    p1 = Point(1, 0, 0)
    p2 = Point([1, 0, 0, 0])
    assert dist(p1, p2) == dist(p2, p1) == np.inf

    p1 = Point(0, 0, 0)
    p2 = Point(1, 0, 0)
    assert np.isclose(dist(p1, p2), 1)

    e = Plane(1, 0, 0, 0)
    assert np.isclose(dist(e, p2), 1)

    p = Point(1, 2, 0)
    l = Line(Point(0, 0, 0), Point(3, 0, 0))
    assert np.isclose(dist(p, l), 2)

    l = Line(p2, Point(1, 1, 0))
    assert np.isclose(dist(l, e), 1)
    assert np.isclose(dist(l, p1), 1)

    p = Point(0, 0)
    poly = Rectangle(Point(-1, 1), Point(1, 1), Point(1, 2), Point(-1, 2))
    assert np.isclose(dist(p, poly), 1)

    p = Point(0, 0, 0)
    poly = Rectangle(Point(-1, -1, 1), Point(1, -1, 1), Point(1, 1, 1), Point(-1, 1, 1))
    assert np.isclose(dist(p, poly), 1)
    assert np.isclose(dist(Point(-1, -1, 0), poly), 1)
    assert np.isclose(dist(Point(-4, 0, -3), poly), 5)

    a = Point(0, 0, 0)
    b = Point(1, 0, 0)
    c = Point(0, 1, 0)
    d = Point(0, 0, 1)
    cube = Cuboid(a, b, c, d)
    # TODO: speed this up
    assert np.isclose(dist(p, cube), 0)
    assert np.isclose(dist(Point(-1, 0, 0), cube), 1)
    assert np.isclose(dist(Point(0.5, 0.5, 2), cube), 1)

    p = PointCollection([(1, 0, 1), (1, 1, 0)], homogenize=True)  # type: ignore[assignment]
    e = PlaneCollection([(0, 0, 1, 0), (1, 0, 0, 0)])  # type: ignore[assignment]
    s = Segment(Point(1, 0, 1), Point(1, 2, 1))
    # TODO: speed this up
    assert np.allclose(dist(e, p), 1)
    assert np.allclose(dist(p, cube), 0)
    assert np.allclose(dist(p, poly), [0, 1])
    assert np.allclose(dist(p, s), [0, 1])


def test_angle() -> None:
    a = Point(0, 0)
    b = Point(1, 1)
    c = Point(1, 0)

    assert np.isclose(angle(a, b, c), np.pi / 4)
    assert np.isclose(angle(a, c, b), -np.pi / 4)

    e1 = Plane(1, 0, 0, 0)
    e2 = Plane(0, 0, 1, 0)

    assert np.isclose(abs(angle(e1, e2)), np.pi / 2)

    p1 = Point(0, 0, 0)
    p2 = Point(0, 1, 0)
    p3 = Point(1, 0, 0)

    assert np.isclose(abs(angle(p1, p2, p3)), np.pi / 2)

    l = Line(p1, p2)
    m = Line(p1, p3)

    assert np.isclose(abs(angle(l, m)), np.pi / 2)

    p1 = PointCollection([(0, 0), (0, 0)], homogenize=True)  # type: ignore[assignment]
    p2 = PointCollection([(1, 1), (0, 1)], homogenize=True)  # type: ignore[assignment]
    p3 = PointCollection([(1, 0), (1, 1)], homogenize=True)  # type: ignore[assignment]

    assert np.allclose(angle(p1, p2, p3), np.pi / 4)

    l = LineCollection(p1, p2)  # type: ignore[assignment]
    m = LineCollection(p1, p3)  # type: ignore[assignment]

    assert np.allclose(angle(l, m), np.pi / 4)

    e1 = PlaneCollection([(0, 0, 1, 0), (1, 0, 0, 0)])  # type: ignore[assignment]
    e2 = PlaneCollection([(1, 1, 1, 0), (0, 1, 0, 0)])  # type: ignore[assignment]

    assert np.allclose(angle(e1, e2), [np.arccos(1 / np.sqrt(3)), np.pi / 2])


def test_angle_bisectors() -> None:
    a = Point(0, 0)
    b = Point(1, 1)
    c = Point(1, 0)
    l = Line(a, b)
    m = Line(a, c)
    q, r = angle_bisectors(l, m)
    assert is_perpendicular(q, r)
    assert np.isclose(angle(l, q), angle(q, m))

    p1 = Point(0, 0, 0)
    p2 = Point(0, 1, 0)
    p3 = Point(1, 0, 0)
    l = Line(p1, p2)
    m = Line(p1, p3)
    q, r = angle_bisectors(l, m)
    assert is_perpendicular(q, r)
    assert np.isclose(angle(l, q), angle(q, m))

    p1 = PointCollection([(0, 0), (0, 0)], homogenize=True)  # type: ignore[assignment]
    p2 = PointCollection([(1, 1), (0, 1)], homogenize=True)  # type: ignore[assignment]
    p3 = PointCollection([(1, 0), (1, 1)], homogenize=True)  # type: ignore[assignment]
    l = LineCollection(p1, p2)  # type: ignore[assignment]
    m = LineCollection(p1, p3)  # type: ignore[assignment]
    q, r = angle_bisectors(l, m)
    assert all(is_perpendicular(q, r))
    assert np.allclose(angle(l, q), angle(q, m))


def test_is_cocircular() -> None:
    p = Point(0, 1)
    t = rotation(np.pi / 3)

    assert is_cocircular(p, t * p, t * t * p, t * t * t * p)  # type: ignore[arg-type]

    p = PointCollection([(0, 1), (1, 0)], homogenize=True)  # type: ignore[assignment]
    t = rotation(np.pi / 3)

    assert all(is_cocircular(p, t * p, t * t * p, t * t * t * p))  # type: ignore[arg-type]


def test_is_coplanar() -> None:
    p1 = Point(1, 1, 0)
    p2 = Point(2, 1, 0)
    p3 = Point(3, 4, 0)
    p4 = Point(0, 2, 0)

    assert is_coplanar(p1, p2, p3, p4)

    p1 = PointCollection([(1, 0), (1, 1)], homogenize=True)  # type: ignore[assignment]
    p2 = PointCollection([(2, 0), (2, 1)], homogenize=True)  # type: ignore[assignment]
    p3 = PointCollection([(3, 0), (3, 1)], homogenize=True)  # type: ignore[assignment]
    p4 = PointCollection([(4, 0), (4, 1)], homogenize=True)  # type: ignore[assignment]

    assert all(is_coplanar(p1, p2, p3, p4))


def test_is_perpendicular() -> None:
    l = Line(0, 1, 0)
    m = Line(1, 0, 0)
    assert is_perpendicular(l, m)

    p1 = Point(0, 0, 0)
    p2 = Point(0, 1, 0)
    p3 = Point(1, 0, 0)
    l = Line(p1, p2)
    m = Line(p1, p3)
    assert is_perpendicular(l, m)

    e1 = Plane(p1, p2, p3)
    e2 = Plane(p1, p2, Point(0, 0, 1))
    assert is_perpendicular(e1, e2)

    l = LineCollection([(0, 1, 0), (0, 1, 0)])  # type: ignore[assignment]
    m = LineCollection([(1, 0, 0), (1, -1, 0)])  # type: ignore[assignment]
    assert list(is_perpendicular(l, m)) == [True, False]

    e1 = PlaneCollection([(0, 0, 1, 0), (1, 0, 0, 0)])  # type: ignore[assignment]
    e2 = PlaneCollection([(0, 1, 0, 0), (0, 0, 1, 0)])  # type: ignore[assignment]
    assert all(is_perpendicular(e1, e2))


def test_pappos() -> None:
    a1 = Point(0, 1)
    b1 = Point(1, 2)
    c1 = Point(2, 3)

    a2 = Point(0, 0)
    b2 = Point(1, 0)
    c2 = Point(2, 0)

    p = a1.join(b2).meet(b1.join(a2))
    q = b1.join(c2).meet(c1.join(b2))
    r = c1.join(a2).meet(a1.join(c2))

    assert is_collinear(p, q, r)


def test_cp1() -> None:
    p = Point(1 + 0j)
    q = Point(0 + 1j)
    m = Transformation([[np.e ** (np.pi / 2 * 1j), 0], [0, 1]])  # type: ignore[arg-type]
    assert m * p == q
    c = crossratio(p, q, m * q, m * m * q)
    assert np.isclose(np.real(c), c)


def test_harmonic_set() -> None:
    a = Point(0, 0)
    b = Point(1, 1)
    c = Point(3, 3)
    d = harmonic_set(a, b, c)
    assert np.isclose(crossratio(a, b, c, d), -1)

    a = Point(0, 0, 0)
    b = Point(1, 1, 0)
    c = Point(3, 3, 0)
    d = harmonic_set(a, b, c)
    assert np.isclose(crossratio(a, b, c, d), -1)

    p1 = PointCollection([(0, 0), (1, 1)], homogenize=True)
    p2 = PointCollection([(1, 1), (2, 1)], homogenize=True)
    p3 = PointCollection([(3, 3), (3, 1)], homogenize=True)
    p4 = harmonic_set(p1, p2, p3)
    assert np.allclose(crossratio(p1, p2, p3, p4), -1)
