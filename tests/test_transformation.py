import numpy as np
from geometer import Point, Line, Transformation, translation, rotation, angle, scaling, reflection


def test_from_points():
    p1 = Point(0, 0)
    p2 = Point(1, 0)
    p3 = Point(0, 1)
    p4 = Point(3, 5)
    l = Line(p1, p3)

    M = Transformation.from_points((p1, p1 + Point(1, 1)), (p2, p2 + Point(1, 1)), (p3, p3 + Point(1, 1)), (p4, p4 + Point(1, 1)))

    assert M*p3 == Point(1, 2)
    assert (M*l).contains(M*p1)
    assert (M*l).contains(M*p3)


def test_translation():
    p = Point(0, 1)
    t = translation(0, -1)
    assert t*p == Point(0, 0)

    l = Line(Point(0, 0, 1), Point(1, 0, 1))
    t = translation(0, 0, -1)
    assert t*l == Line(Point(0, 0, 0), Point(1, 0, 0))


def test_inverse():
    E = Transformation(np.eye(4))
    M = rotation(np.pi, axis=Point(0, 1, 0))
    assert M.inverse()*M == E


def test_pow():
    t = translation(1, 2)

    assert t**0 == Transformation(np.eye(3))
    assert t**1 == t
    assert t**2 == translation(2, 4)
    assert t**3 == translation(3, 6)
    assert t**(-2) == translation(-2, -4)


def test_rotation():
    p = Point(0, 1)
    t = rotation(-np.pi)
    assert t*p == Point(0, -1)

    p = Point(1, 0, 0)
    t = rotation(-np.pi/2, axis=Point(0, 0, 1))
    assert t * p == Point(0, 1, 0)

    p = Point(-1, 1, 0)
    a = np.pi / 7
    t = rotation(a, axis=Point(1, 1, 2))
    assert np.isclose(angle(p, t * p), a)


def test_scaling():
    p = Point(1, 1, 2)
    s = scaling(3, -4.5, 5)

    assert s * p == Point(3, -4.5, 10)


def test_reflection():
    p = Point(-1, 1)
    r = reflection(Line(1, -1, 1))

    assert r * p == Point(0, 0)
