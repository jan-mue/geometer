import numpy as np
from .point import Point, Line, Plane, I, J, infty, infty_plane, join, meet
from .curve import Conic, Circle, absolute_conic
from .base import LeviCivitaTensor, TensorDiagram
from .exceptions import IncidenceError, NotCollinear


geometries = {
    "euclidean": (1, 1/2j, Conic([[0, 0, 0], [0, 0, 0], [0, 0, 1]]), Conic([[1, 0, 0], [0, 1, 0], [0, 0, 0]], True)),
    "hyperbolic": (-1/2, 1/2j, Circle(), Circle().dual),
    "elliptic": (1/2j, 1/2j, absolute_conic, absolute_conic.dual)
}


def crossratio(a, b, c, d, from_point=None):
    """Calculates the cross ratio of points or lines.

    Parameters
    ----------
    a : :obj:`Point` or :obj:`Line`
        A point or line.
    b : :obj:`Point` or :obj:`Line`
        A point or line.
    c : :obj:`Point` or :obj:`Line`
        A point or line.
    d : :obj:`Point` or :obj:`Line`
        A point or line.
    from_point : Point, optional
        A 2D point, only possible if the other arguments are also 2D points.

    Returns
    -------
    :obj:`float` of :obj:`complex`
        The cross ration of the given objects.

    """

    if a == b:
        return 1

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
    """Constructs a fourth point that forms a harmonic set with the given points.

    Parameters
    ----------
    a : Point
        A point in 2D or 3D.
    b : Point
        A point in 2D or 3D.
    c : Point
        A point in 2D or 3D.

    Returns
    -------
    Point
        The point that forms a harmonic set with the given points.

    """
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


def angle(*args, geometry="euclidean"):
    """Calculates the (oriented) angle between given points, lines or planes.

    The function uses the Laguerre formula to calculate angles in two or three dimensional projective space
    using cross ratios. To calculate the cross ratio of planes, two additional planes tangent to the absolute
    conic are constructed.

    Parameters
    ----------
    *args
        The objects between which the function calculates the angle. This can be 2 or 3 points, 2 lines or 2 planes.
    geometry : str or tuple, optional
        The geometry to use for the calculation, default is euclidean. Can be a tuple of two constants and two primal
        dual conics, i.e. (c_dist, c_ang, A, B) with A*B being a multiple of the identity.

    Returns
    -------
    float
        The oriented angle between the given objects.

    """

    if isinstance(geometry, str):
        geometry = geometries[geometry]

    c1, c2, A, B = geometry

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

    i = B.intersect(a)
    x, y = 2*i if len(i) == 1 else i
    x, y = infty.meet(x), infty.meet(y)

    return c2*np.log(crossratio(b, c, x, y, a))


def angle_bisectors(l, m):
    """Constructs the angle bisectors of two given lines.

    Parameters
    ----------
    l : Line
        A line in 2D or 3D.
    m : Line
        A line in 2D or 3D.

    Returns
    -------
    :obj:`tuple` of :obj:`Line`
        The two angle bisectors.

    """
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


def dist(p, q, geometry="euclidean"):
    """Calculates the distance between two objects.

    Parameters
    ----------
    p : :obj:`Point`, :obj:`Line` or :obj:`Plane`
        A point, line or plane to calculate the distance to.
    q : :obj:`Point`, :obj:`Line` or :obj:`Plane`
        A point, line or plane to calculate the distance to.
    geometry : str or tuple, optional
        The geometry to use for the calculation, default is euclidean. Can be a tuple of two constants and two primal
        dual conics, i.e. (c_dist, c_ang, A, B) with A*B being a multiple of the identity.

    Returns
    -------
    float
        The distance between the given objects.

    """
    if isinstance(p, (Plane, Line)) and isinstance(q, Point):
        return dist(p.project(q), q)
    if isinstance(p, Point) and isinstance(q, (Plane, Line)):
        return dist(q.project(p), p)
    if isinstance(p, Plane) and isinstance(q, (Plane, Line)):
        return dist(p, Point(q.basis_matrix[0, :]))
    if isinstance(q, Plane) and isinstance(p, Line):
        return dist(q, p.base_point)

    if isinstance(geometry, str):
        geometry = geometries[geometry]

    c1, c2, A, B = geometry

    if p.dim == 3:
        l = Line(p, q)
        a, b = p.normalized_array, q.normalized_array

        for r in np.eye(4):
            if not l.contains(Point(r)):
                break

        m, _ = np.linalg.qr(np.array([r, a, b]).T)
        m = m.T[np.argsort(np.abs(m.T.dot(a)))]
        p, q = Point(m.dot(a)), Point(m.dot(b))

    l = Line(p, q)
    i = A.intersect(l)

    if len(i) == 1:
        # parabolic measurement -> use euclidean distance
        a, b = p.normalized_array, q.normalized_array

        pqi = np.linalg.det([a, b, I.array])
        pqj = np.linalg.det([a, b, J.array])
        return np.sqrt(pqi * pqj)
    else:
        x, y = i

    return c1*np.log(crossratio(p, q, x, y))


def is_cocircular(a, b, c, d):
    """Tests whether four points lie on a circle.

    Parameters
    ----------
    a : Point
        A point in RP2 or CP1.
    b : Point
        A point in RP2 or CP1.
    c : Point
        A point in RP2 or CP1.
    d : Point
        A point in RP2 or CP1.

    Returns
    -------
    bool
        True if the four points lie on a circle.

    """
    if a.dim == 1:
        return np.isreal(crossratio(a, b, c, d))
    else:
        i = crossratio(a, b, c, d, I)
        j = crossratio(a, b, c, d, J)
        return np.isclose(i, j)


def is_perpendicular(l, m):
    """Tests whether two lines are perpendicular.

    Parameters
    ----------
    l : Line
        A line in 2D or 3D.
    m : Line
        A line in 2D or 3D.

    Returns
    -------
    bool
        True if the two lines are perpendicular.

    """
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
    """Tests whether the given points or lines are collinear, coplanar or concurrent. Works in any dimension.

    Due to line point duality this function has dual versions :obj:`is_collinear`, :obj:`is_collinear` and
    :obj:`is_concurrent`.

    Parameters
    ----------
    *args
        The points or lines to test.

    Returns
    -------
    bool
        True if the given points are coplanar (in 3D) or collinear (in 2D) or if the given lines are concurrent.

    """
    n = args[0].dim + 1
    if not np.isclose(np.linalg.det([a.array for a in args[:n]]), 0):
        return False
    if len(args) == n:
        return True
    covariant = args[0].tensor_shape[1] > 0
    e = LeviCivitaTensor(n, covariant=covariant)
    tensor = TensorDiagram(*[(e, a) if covariant else (a, e) for a in args[:n-1]])
    for t in args[n:]:
        if not (covariant and t*tensor == 0 or not covariant and tensor*t == 0):
            return False
    return True


is_collinear = is_coplanar
is_concurrent = is_coplanar
