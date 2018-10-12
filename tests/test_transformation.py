import numpy as np
from geometer import Point, Line, Transformation, translation, rotation


def test_from_points():
    p1 = Point(0,0)
    p2 = Point(1,0)
    p3 = Point(0,1)
    p4 = Point(3,5)
    l = Line(p1, p3)

    M = Transformation.from_points((p1, p1 + Point(1, 1)), (p2, p2 + Point(1, 1)), (p3, p3 + Point(1, 1)), (p4, p4 + Point(1, 1)))

    assert M*p3 == Point(1,2)
    assert (M*l).contains(M*p1)
    assert (M*l).contains(M*p3)


def test_translation():
    p = Point(0,1)
    t = translation(0,-1)
    assert t*p == Point(0,0)


def test_rotation():
    p = Point(0, 1)
    t = rotation(-np.pi)
    assert t*p == Point(0,-1)
