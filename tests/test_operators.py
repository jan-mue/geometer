import numpy as np
from geometer import *


def test_is_collinear():
    p1 = Point(1,0)
    p2 = Point(2,0)
    p3 = Point(3,0)
    l = Line(p1, p2)
    assert l.contains(p3)
    assert is_collinear(p1, p2, p3)


def test_dist():
    p = Point(0,0)
    q = Point(1,0)
    assert dist(p, q) == 1


def test_angle():
    a = Point(0, 0)
    b = Point(1, 1)
    c = Point(1,0)
    assert np.isclose(angle(a,b,c), np.pi/4)

def test_is_cocircular():
    p = Point(0,1)
    t = rotation(np.pi/3)

    assert is_cocircular(p, t*p, t*t*p, t*t*t*p)


def test_is_perpendicular():
    l = Line(0,1,0)
    m = Line(1,0,0)
    assert is_perpendicular(l,m)


def test_pappos():
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


def test_cp1():
    p = Point(1+0j)
    q = Point(0+1j)
    m = Transformation([[np.e**(np.pi/2*1j), 0], [0, 1]])
    assert m*p == q
    c = crossratio(p,q,m*q,m*m*q)
    assert np.isclose(np.real(c), c)