from geometer import Point, Line, Plane, join, meet, is_collinear


def test_collinear():
    p1 = Point(1,0)
    p2 = Point(2,0)
    p3 = Point(3,0)
    l = Line(p1, p2)
    assert l.contains(p3)
    assert is_collinear(p1, p2, p3)

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

def test_add():
    p = Point(1,0)
    q = Point(0,1)
    assert p + q == Point(1,1)

def test_parallel():
    p = Point(0,1)
    q = Point(1,1)
    r = Point(0,0)
    l = Line(p, q)
    m = l.parallel(through=r)
    assert m == Line(0,1,0)

def test_3d():
    p1 = Point(1,1,0)
    p2 = Point(2,1,0)
    p3 = Point(3,4,0)
    p4 = Point(0,2,0)
    assert join(p1,p2,p3).contains(p4)
    assert is_collinear(p1, p2, p3, p4)
