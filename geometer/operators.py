from itertools import product
import numpy as np
from .point import Point, Line, Plane, I, J, infty, infty_plane
from .curve import absolute_conic
from .exceptions import IncidenceError, NotCollinear, LinearDependenceError


def crossratio(a, b, c, d, from_point=None):
    if isinstance(a, Line):
        if not is_concurrent(a,b,c,d):
            raise IncidenceError("The lines are not concurrent: " + str([a,b,c,d]))
        from_point = a.meet(b)
        a, b, c, d = a.base_point, b.base_point, c.base_point, d.base_point

    if isinstance(a, Plane):
        l = a.meet(b)
        e = Plane(l.direction.array)
        a, b, c, d = e.meet(a), e.meet(b), e.meet(c), e.meet(d)
        m = e.basis_matrix
        p = e.meet(l)
        from_point = Point(m.dot(p.array))
        a = Point(m.dot((p + a.direction).array))
        b = Point(m.dot((p + b.direction).array))
        c = Point(m.dot((p + c.direction).array))
        d = Point(m.dot((p + d.direction).array))

    if from_point is None and len(a.array) > 2:
        if not is_collinear(a,b,c,d):
            raise NotCollinear("The points are not collinear: " + str([a, b, c, d]))
        l = a.join(b)
        a = Point(l.basic_coeffs(a))
        b = Point(l.basic_coeffs(b))
        c = Point(l.basic_coeffs(c))
        d = Point(l.basic_coeffs(d))

    o = (from_point or []) and [from_point.array]
    ac = np.linalg.det(o + [a.array, c.array])
    bd = np.linalg.det(o + [b.array, d.array])
    ad = np.linalg.det(o + [a.array, d.array])
    bc = np.linalg.det(o + [b.array, c.array])

    return ac * bd / (ad * bc)


def angle(*args):
    if len(args) == 3:
        a, b, c = args
        if a.dim == 3:
            e = Plane(*args)
            basis = e.basis_matrix
            a, b, c = Point(basis.dot(a.array)), Point(basis.dot(b.array)), Point(basis.dot(c.array))

    elif len(args) == 2:
        if isinstance(args[0], Plane):
            e, f = args
            l = e.meet(f)
            p = l.meet(infty_plane)
            polar = Line(p.array[:-1])
            tangent_points = absolute_conic.intersect(polar)
            tangent_points = [Point(np.append(p.array, 0)) for p in tangent_points]
            i = l.join(p.join(tangent_points[0]))
            j = l.join(p.join(tangent_points[1]))
            return 1/2j*np.log(crossratio(e, f, i, j))

        l, m = args
        a = l.meet(m)

        if a.dim == 3:
            e = Plane(l, m)
            basis = e.basis_matrix
            a = Point(basis.dot(a.array))
            b = Point(basis.dot(l.meet(infty_plane).array))
            c = Point(basis.dot(m.meet(infty_plane).array))
        else:
            b = l.meet(infty)
            c = m.meet(infty)
    else:
        raise ValueError("Expected 2 or 3 arguments, got %s." % len(args))

    return 1/2j*np.log(crossratio(b,c, I,J, a))


def angle_bisectors(l, m):
    o = l.meet(m)
    if o.dim == 3:
        e = Plane(l, m)
        basis = e.basis_matrix
        L = Point(basis.dot(l.meet(infty_plane).array))
        M = Point(basis.dot(m.meet(infty_plane).array))
    else:
        L, M = l.meet(infty), m.meet(infty)
    p = Point(0, 0)
    li = np.linalg.det([p.array, L.array, I.array])
    lj = np.linalg.det([p.array, L.array, J.array])
    mi = np.linalg.det([p.array, M.array, I.array])
    mj = np.linalg.det([p.array, M.array, J.array])
    a, b = np.sqrt(lj*mj), np.sqrt(li*mi)
    r, s = a*I+b*J, a*I-b*J
    if o.dim == 3:
        r, s = Point(basis.T.dot(r.array)), Point(basis.T.dot(s.array))
    return Line(o, r), Line(o, s)


def dist(p, q):
    # TODO: lines & planes
    a, b = p.array, q.array
    if p.dim == 3:
        for i in range(4):
            try:
                r = np.zeros(4)
                r[3-i] = 1
                e = Plane(p, q, Point(r))
            except LinearDependenceError:
                continue
            else:
                break

        m = e.basis_matrix
        a, b = m.dot(a), m.dot(b)

    pqi = np.linalg.det([a, b, I.array])
    pqj = np.linalg.det([a, b, J.array])
    return np.sqrt(pqi*pqj)


def intersection(*args):
    result = []
    for x, y in product(args, args):
        for i in x.intersect(y):
            if i not in result:
                result.append(i)
    return result


def is_cocircular(a,b,c,d):
    if np.any(np.iscomplex([a.array, b.array, c.array, d.array])) :
        cross = crossratio(a,b,c,d)
        return np.isclose(np.real(cross), cross)
    else:
        i = crossratio(a,b,c,d, I)
        j = crossratio(a,b,c,d, J)
        return np.isclose(i, j)


def is_perpendicular(l, m):
    if l.dim == 3:
        e = Plane(l, m)
        basis = e.basis_matrix
        L = Point(basis.dot(l.meet(infty_plane).array))
        M = Point(basis.dot(m.meet(infty_plane).array))
    else:
        L = l.meet(infty)
        M = m.meet(infty)
    return np.isclose(crossratio(L,M, I,J, Point(1, 1)), -1)


def is_collinear(a,b,c, *args):
    if not np.isclose(np.linalg.det([a.array, b.array, c.array]), 0):
        return False
    n = np.cross(a.array, b.array)
    for d in args:
        if not np.isclose(np.vdot(n, d.array), 0):
            return False
    return True


def is_coplanar(a, b, c, d, *args):
    if not np.isclose(np.linalg.det([a.array, b.array, c.array, d.array]), 0):
        return False
    p = Plane(b, c, d)
    for e in args:
        if not p.contains(e):
            return False
    return True


is_concurrent = is_collinear