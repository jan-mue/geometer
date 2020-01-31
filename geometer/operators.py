import numpy as np
from .point import Point, Line, Plane, I, J, infty, infty_plane, join, meet
from .curve import absolute_conic
from .base import LeviCivitaTensor, TensorDiagram
from .exceptions import IncidenceError, NotCollinear
from .utils import orth


def crossratio(a, b, c, d, from_point=None):
    """Calculates the cross ratio of points or lines.

    Parameters
    ----------
    a, b, c, d : Point, Line or Plane
        The points, lines or planes (any dimension) to calculate the cross ratio of.
    from_point : Point, optional
        A 2D point, only possible if the other arguments are also 2D points.

    Returns
    -------
    float or complex
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

    with np.errstate(divide="ignore"):
        return ac * bd / (ad * bc)


def harmonic_set(a, b, c):
    """Constructs a fourth point that forms a harmonic set with the given points.

    The three given points must be collinear.

    If the returned point is d, the points {{a, b}, {c, d}} will be in harmonic position.

    Parameters
    ----------
    a, b, c : Point
        The points (any dimension) that are used to construct the fourth point in the harmonic set.

    Returns
    -------
    Point
        The point that forms a harmonic set with the given points.

    """
    l = Line(a, b)
    o = l.general_point
    n = l.dim + 1

    if n > 3:
        e = join(l, o)
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
    """Calculates the (oriented) angle between given points, lines or planes.

    The function uses the Laguerre formula to calculate angles in two or three dimensional projective space
    using cross ratios. To calculate the angle between two planes, two additional planes tangent to the absolute
    conic are constructed (see [1]).

    Since the Laguerre formula uses the complex logarithm (which gives values between -pi*i and pi*i) and multiplies
    it with 1/2i, this function can only calculate angles between -pi/2 and pi/2.

    The sign of the angle is determined by the order of the arguments. The points passed to the cross ratio are in
    the same order as the arguments to this function.
    When three points are given as arguments, the first point is the point at which the angle is calculated.

    Parameters
    ----------
    *args
        The objects between which the function calculates the angle. This can be 2 or 3 points, 2 lines or 2 planes.

    Returns
    -------
    float
        The oriented angle between the given objects.

    References
    ----------
    .. [1] Olivier Faugeras, Three-dimensional Computer Vision, Page 30

    """
    if len(args) == 3:
        a, b, c = args
        if a.dim > 2:
            e = join(*args)
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

        if a.dim > 2:
            e = join(x, y)
            basis = e.basis_matrix
            a = Point(basis.dot(a.array))
            b = Point(basis.dot(x.meet(infty_plane).array))
            c = Point(basis.dot(y.meet(infty_plane).array))
        else:
            b = x.meet(infty)
            c = y.meet(infty)
    else:
        raise ValueError("Expected 2 or 3 arguments, got %s." % len(args))

    return np.real(1/2j*np.log(crossratio(b, c, I, J, a)))


def angle_bisectors(l, m):
    """Constructs the angle bisectors of two given lines.

    Parameters
    ----------
    l, m : Line
        Two lines in any dimension.

    Returns
    -------
    tuple of Line
        The two angle bisectors.

    """
    o = l.meet(m)

    if o.dim > 2:
        e = join(l, m)
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

    if o.dim > 2:
        r, s = Point(basis.T.dot(r.array)), Point(basis.T.dot(s.array))

    return Line(o, r), Line(o, s)


def dist(p, q):
    """Calculates the (euclidean) distance between two objects.

    Parameters
    ----------
    p, q : Point, Line or Plane
        The points, lines or planes to calculate the distance between.

    Returns
    -------
    float
        The distance between the given objects.

    """
    if p == q:
        return 0

    if isinstance(p, (Plane, Line)) and isinstance(q, Point):
        return dist(p.project(q), q)
    if isinstance(p, Point) and isinstance(q, (Plane, Line)):
        return dist(q.project(p), p)
    if isinstance(p, Plane) and isinstance(q, (Plane, Line)):
        return dist(p, Point(q.basis_matrix[0, :]))
    if isinstance(q, Plane) and isinstance(p, Line):
        return dist(q, p.base_point)

    if p.dim > 2:
        x = np.array([p.normalized_array, q.normalized_array])
        z = x[:, -1]

        m = orth(x.T)
        x = m.T.dot(x.T)
        x = np.append(x, [z], axis=0).T
        p, q = Point(x[0]), Point(x[1])

    pqi = np.linalg.det([p.array, q.array, I.array])
    pqj = np.linalg.det([p.array, q.array, J.array])
    pij = np.linalg.det([p.array, I.array, J.array])
    qij = np.linalg.det([q.array, I.array, J.array])

    with np.errstate(divide="ignore", invalid="ignore"):
        return 4*abs(np.sqrt(pqi * pqj)/(pij*qij))


def is_cocircular(a, b, c, d, rtol=1.e-5, atol=1.e-8):
    """Tests whether four points lie on a circle.

    Parameters
    ----------
    a, b, c, d : Point
        Four points in RP2 or CP1.
    rtol : float, optional
        The relative tolerance parameter.
    atol : float, optional
        The absolute tolerance parameter.

    Returns
    -------
    bool
        True if the four points lie on a circle.

    """
    if a.dim == 1:
        return np.isreal(crossratio(a, b, c, d))

    elif a.dim > 2:
        e = join(a, b, c)
        basis = e.basis_matrix
        a = Point(basis.dot(a.array))
        b = Point(basis.dot(b.array))
        c = Point(basis.dot(c.array))
        d = Point(basis.dot(d.array))

    i = crossratio(a, b, c, d, I)
    j = crossratio(a, b, c, d, J)
    return np.isclose(i, j, rtol, atol)


def is_perpendicular(l, m, rtol=1.e-5, atol=1.e-8):
    """Tests whether two lines/planes are perpendicular.

    Parameters
    ----------
    l, m : Line or Plane
        Two lines in any dimension or two planes in 3D.
    rtol : float, optional
        The relative tolerance parameter.
    atol : float, optional
        The absolute tolerance parameter.

    Returns
    -------
    bool
        True if the two lines/planes are perpendicular.

    """
    if l.dim < 3:
        L = l.meet(infty)
        M = m.meet(infty)

    elif isinstance(l, Line) and isinstance(m, Line):
        e = join(l, m)
        basis = e.basis_matrix
        L = Point(basis.dot(l.meet(infty_plane).array))
        M = Point(basis.dot(m.meet(infty_plane).array))

    elif isinstance(l, Plane) and isinstance(m, Plane):
        x = l.meet(m)
        p = x.meet(infty_plane)
        polar = Line(p.array[:-1])
        tangent_points = absolute_conic.intersect(polar)
        tangent_points = [Point(np.append(p.array, 0)) for p in tangent_points]
        i = x.join(p.join(tangent_points[0]))
        j = x.join(p.join(tangent_points[1]))
        return np.isclose(crossratio(l, m, i, j), -1, rtol, atol)

    else:
        raise NotImplementedError("Only two lines or two planes are supported.")

    return np.isclose(crossratio(L, M, I, J, Point(1, 1)), -1, rtol, atol)


def is_coplanar(*args, tol=1.e-8):
    """Tests whether the given points or lines are collinear, coplanar or concurrent. Works in any dimension.

    Due to line point duality this function has dual versions :obj:`is_collinear`, :obj:`is_collinear` and
    :obj:`is_concurrent`.

    Parameters
    ----------
    *args
        The points or lines to test.
    tol : float, optional
            The accepted tolerance.

    Returns
    -------
    bool
        True if the given points are coplanar (in 3D) or collinear (in 2D) or if the given lines are concurrent.

    """
    n = args[0].dim + 1
    if not np.isclose(np.linalg.det([a.array for a in args[:n]]), 0, atol=tol):
        return False
    if len(args) == n:
        return True
    covariant = args[0].tensor_shape[1] > 0
    e = LeviCivitaTensor(n, covariant=covariant)
    diagram = TensorDiagram(*[(e, a) if covariant else (a, e) for a in args[:n-1]])
    tensor = diagram.calculate()
    for t in args[n:]:
        x = t*tensor if covariant else tensor*t
        if not np.isclose(x.array, 0, atol=tol):
            return False
    return True


is_collinear = is_coplanar
is_concurrent = is_coplanar
