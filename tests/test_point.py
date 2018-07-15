from geometer import Point, Line, Plane, join, meet, is_collinear


def test_collinear():
    p1 = Point(1,0)
    p2 = Point(2,0)
    p3 = Point(3,0)
    l = Line(p1, p2)
    assert l.contains(p3)
    assert is_collinear(p1, p2, p3)

def test_add():
    p = Point(1,0)
    q = Point(0,1)
    assert p + q == Point(1,1)

def test_3d():
    p1 = Point(1,1,0, dimension=3)
    p2 = Point(2,1,0, dimension=3)
    p3 = Point(3,4,0, dimension=3)
    p4 = Point(0,2,0, dimension=3)
    assert join(p1,p2,p3).contains(p4)
    assert is_collinear(p1, p2, p3, p4)
