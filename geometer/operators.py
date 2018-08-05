import numpy as np
from .point import Point, I, J, infty


def crossratio(a, b, c, d, from_point=None):
    o = (from_point or []) and [from_point.array]
    ac = np.linalg.det(o + [a.array, c.array])
    bd = np.linalg.det(o + [b.array, d.array])
    ad = np.linalg.det(o + [a.array, d.array])
    bc = np.linalg.det(o + [b.array, c.array])

    return ac * bd / (ad * bc)


def angle(*args):
    if len(args) == 3:
        a,b,c = args
    elif len(args) == 2:
        l, m = args
        a = l.meet(m)
        b = l.meet(infty)
        c = m.meet(infty)
    else:
        raise ValueError("Expected 2 or 3 arguments, got %s." % len(args))

    return 1/2j*np.log(crossratio(b,c, I,J, a))

def dist(p, q):
    pqi = np.linalg.det([p.array, q.array, I.array])
    pqj = np.linalg.det([p.array, q.array, J.array])
    return np.sqrt(pqi*pqj)

def is_cocircular(a,b,c,d):
    if np.any(np.iscomplex([a.array, b.array, c.array, d.array])) :
        cross = crossratio(a,b,c,d)
        return np.isclose(np.real(cross), cross)
    else:
        i = crossratio(a,b,c,d, I)
        j = crossratio(a,b,c,d, J)
        return np.isclose(i, j)

def is_perpendicular(l, m):
    L = l.meet(infty)
    M = m.meet(infty)
    return np.isclose(crossratio(L,M, I,J, Point(1,1)), -1)


def is_collinear(*args):
    return np.isclose(np.linalg.det([p.array for p in args]), 0)


is_concurrent = is_collinear