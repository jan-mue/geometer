import numpy as np
from .point import Point, Line, Plane, I, J, infty, infty_plane, join, meet
from .curve import absolute_conic
from .base import LeviCivitaTensor, TensorDiagram
from .exceptions import IncidenceError, NotCollinear


def crossratio(a, b, c, d, from_point=None):
    if isinstance(a, Line):
        if not is_concurrent(a, b, c, d):
            raise IncidenceError("The lines are not concurrent: " + str([a, b, c, d]))

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

    if a.dim > 2 or (from_point is None and a.dim == 2):

        if not is_collinear(a, b, c, d):
            raise NotCollinear("The points are not collinear: " + str([a, b, c, d]))

        l = a.join(b)
        basis = l.basis_matrix
        a = Point(basis.dot(a.array))
        b = Point(basis.dot(b.array))
        c = Point(basis.dot(c.array))
        d = Point(basis.dot(d.array))

    o = (from_point or []) and [from_point.array]
    ac = np.linalg.det(o + [a.array, c.array])
    bd = np.linalg.det(o + [b.array, d.array])
    ad = np.linalg.det(o + [a.array, d.array])
    bc = np.linalg.det(o + [b.array, c.array])

    return ac * bd / (ad * bc)


def harmonic_set(a, b, c):
    l = Line(a, b)
    n = l.dim + 1
    arr = np.zeros(n)
    for i in range(n):
        arr[-i-1] = 1
        o = Point(arr)
        if not l.contains(o):
            break
    if n > 3:
        e = Plane(l, o)
        basis = e.basis_matrix
        a = Point(basis.dot(a.array))
        b = Point(basis.dot(b.array))
        c = Point(basis.dot(c.array))
        o = Point(basis.dot(o.array))
        l = Line(a, b)

    m = Line(o, c)
    p = o + 1/2*m.direction
    result = l.meet(join(meet(o.join(a), p.join(b)), meet(o.join(b), p.join(a))))
    if n > 3:
        return Point(basis.T.dot(result.array))
    return result


def angle(*args):
    if len(args) == 3:
        a, b, c = args
        if a.dim == 3:
            e = Plane(*args)
            basis = e.basis_matrix
            a, b, c = Point(basis.dot(a.array)), Point(basis.dot(b.array)), Point(basis.dot(c.array))

    elif len(args) == 2:
        x, y = args

        if isinstance(x, Plane) and isinstance(y, Plane):
            l = x.meet(y)
            p = l.meet(infty_plane)
            polar = Line(p.array[:-1])
            tangent_points = absolute_conic.intersect(polar)
            tangent_points = [Point(np.append(p.array, 0)) for p in tangent_points]
            i = l.join(p.join(tangent_points[0]))
            j = l.join(p.join(tangent_points[1]))
            return 1/2j*np.log(crossratio(x, y, i, j))

        if isinstance(x, Line) and isinstance(y, Line):
            a = x.meet(y)
        else:
            a = Point(*(x.dim * [0]))
            if isinstance(x, Point):
                x = a.join(x)
            if isinstance(y, Point):
                y = a.join(y)

        if a.dim == 3:
            e = Plane(x, y)
            basis = e.basis_matrix
            a = Point(basis.dot(a.array))
            b = Point(basis.dot(x.meet(infty_plane).array))
            c = Point(basis.dot(y.meet(infty_plane).array))
        else:
            b = x.meet(infty)
            c = y.meet(infty)
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
    if isinstance(p, (Plane, Line)) and isinstance(q, Point):
        return dist(p.project(q), q)
    if isinstance(p, Point) and isinstance(q, (Plane, Line)):
        return dist(q.project(p), p)
    if isinstance(p, Plane) and isinstance(q, (Plane, Line)):
        return dist(p, Point(q.basis_matrix[0, :]))
    if isinstance(q, Plane) and isinstance(p, Line):
        return dist(q, p.base_point)

    a, b = p.normalized_array, q.normalized_array
    if p.dim == 3:
        l = Line(p, q)
        for r in np.eye(4):
            if not l.contains(Point(r)):
                break

        m, _ = np.linalg.qr(np.array([r, a, b]).T)
        a, b = m.T.dot(a), m.T.dot(b)

    pqi = np.linalg.det([a, b, I.array])
    pqj = np.linalg.det([a, b, J.array])
    return np.sqrt(pqi*pqj)


def is_cocircular(a, b, c, d):
    if a.dim == 1:
        return np.isreal(crossratio(a, b, c, d))
    else:
        i = crossratio(a, b, c, d, I)
        j = crossratio(a, b, c, d, J)
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


def is_coplanar(*args):
    n = args[0].dim + 1
    if not np.isclose(np.linalg.det([a.array for a in args[:n]]), 0):
        return False
    if len(args) == n:
        return True
    covariant = args[0].tensor_shape[1] > 0
    e = LeviCivitaTensor(n, covariant=covariant)
    diagram = TensorDiagram(*[(e, a) if covariant else (a, e) for a in args[:n-1]])
    tensor = diagram.calculate()
    for t in args[n:]:
        if not (covariant and t*tensor == 0 or not covariant and tensor*t == 0):
            return False
    return True


is_collinear = is_coplanar
is_concurrent = is_coplanar
