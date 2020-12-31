import numpy as np
from .point import (
    Point,
    Line,
    Plane,
    I,
    J,
    PointCollection,
    LineCollection,
    PlaneCollection,
    infty,
    infty_plane,
    join,
    meet,
)
from .curve import absolute_conic
from .base import LeviCivitaTensor, TensorDiagram, EQ_TOL_REL, EQ_TOL_ABS
from .exceptions import IncidenceError, NotCollinear
from .utils import orth, det, matvec


def crossratio(a, b, c, d, from_point=None):
    """Calculates the cross ratio of points or lines.

    Parameters
    ----------
    a, b, c, d : Point, PointCollection, Line, LineCollection, Plane or PlaneCollection
        The points, lines or planes (any dimension) to calculate the cross ratio of.
    from_point : Point or PointCollection, optional
        A 2D point, only accepted if the other arguments are also 2D points.

    Returns
    -------
    float or complex
        The cross ration of the given objects.

    """

    if a == b:
        return np.ones(a.shape[: len(a._collection_indices)])

    if isinstance(a, (Line, LineCollection)):
        if not np.all(is_concurrent(a, b, c, d)):
            raise IncidenceError("The lines are not concurrent: " + str([a, b, c, d]))

        from_point = a.meet(b)
        a, b, c, d = a.base_point, b.base_point, c.base_point, d.base_point

    if isinstance(a, Plane):
        l = a.meet(b)
        e = Plane(l.direction.array, copy=False)
        a, b, c, d = e.meet(a), e.meet(b), e.meet(c), e.meet(d)
        m = e.basis_matrix
        p = e.meet(l)
        from_point = Point(m.dot(p.array), copy=False)
        a = Point(m.dot((p + a.direction).array), copy=False)
        b = Point(m.dot((p + b.direction).array), copy=False)
        c = Point(m.dot((p + c.direction).array), copy=False)
        d = Point(m.dot((p + d.direction).array), copy=False)

    if isinstance(a, PlaneCollection):
        l = a.meet(b)
        e = PlaneCollection(l.direction.array, copy=False)
        a, b, c, d = e.meet(a), e.meet(b), e.meet(c), e.meet(d)
        m = e.basis_matrix
        p = e.meet(l)
        from_point = PointCollection(m.dot(p.array), copy=False)
        a = PointCollection(m.dot((p + a.direction).array), copy=False)
        b = PointCollection(m.dot((p + b.direction).array), copy=False)
        c = PointCollection(m.dot((p + c.direction).array), copy=False)
        d = PointCollection(m.dot((p + d.direction).array), copy=False)

    if a.dim > 2 or (from_point is None and a.dim == 2):

        if not np.all(is_collinear(a, b, c, d)):
            raise NotCollinear("The points are not collinear: " + str([a, b, c, d]))

        basis = np.stack([a.array, b.array], axis=-2)
        a = matvec(basis, a.array)
        b = matvec(basis, b.array)
        c = matvec(basis, c.array)
        d = matvec(basis, d.array)
        o = []

    elif from_point is not None:
        a, b, c, d, from_point = np.broadcast_arrays(
            a.array, b.array, c.array, d.array, from_point.array
        )
        o = [from_point]
    else:
        a, b, c, d = np.broadcast_arrays(a.array, b.array, c.array, d.array)
        o = []

    ac = det(np.stack(o + [a, c], axis=-2))
    bd = det(np.stack(o + [b, d], axis=-2))
    ad = det(np.stack(o + [a, d], axis=-2))
    bc = det(np.stack(o + [b, c], axis=-2))

    with np.errstate(divide="ignore"):
        return ac * bd / (ad * bc)


def harmonic_set(a, b, c):
    """Constructs a fourth point that forms a harmonic set with the given points.

    The three given points must be collinear.

    If the returned point is d, the points {{a, b}, {c, d}} will be in harmonic position.

    Parameters
    ----------
    a, b, c : Point or PointCollection
        The points (any dimension) that are used to construct the fourth point in the harmonic set.

    Returns
    -------
    Point or PointCollection
        The point that forms a harmonic set with the given points.

    """
    l = join(a, b)
    o = l.general_point
    n = l.dim + 1

    if n > 3:
        e = join(l, o)
        basis = e.basis_matrix
        if isinstance(a, Point):
            a = Point(basis.dot(a.array), copy=False)
            b = Point(basis.dot(b.array), copy=False)
            c = Point(basis.dot(c.array), copy=False)
            o = Point(basis.dot(o.array), copy=False)
        else:
            a = PointCollection(matvec(basis, a.array), copy=False)
            b = PointCollection(matvec(basis, b.array), copy=False)
            c = PointCollection(matvec(basis, c.array), copy=False)
            o = PointCollection(matvec(basis, o.array), copy=False)

        l = join(a, b)

    m = join(o, c)
    p = o + 1 / 2 * m.direction
    result = l.meet(join(meet(o.join(a), p.join(b)), meet(o.join(b), p.join(a))))

    if n > 3:
        if isinstance(a, Point):
            return Point(basis.T.dot(result.array), copy=False)
        return PointCollection(
            matvec(basis, result.array, transpose_a=True), copy=False
        )

    return result


def angle(*args):
    r"""Calculates the (oriented) angle between given points, lines or planes.

    The function uses the Laguerre formula to calculate angles in two or three dimensional projective space
    using cross ratios. To calculate the angle between two planes, two additional planes tangent to the absolute
    conic are constructed (see [1]).

    Since the Laguerre formula uses the complex logarithm (which gives values between :math:`-\pi i` and :math:`\pi i`)
    and multiplies it with :math:`1/2i`, this function can only calculate angles between :math:`-\pi / 2` and
    :math:`\pi / 2`.

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
            a = Point(basis.dot(a.array), copy=False)
            b = Point(basis.dot(b.array), copy=False)
            c = Point(basis.dot(c.array), copy=False)

    elif len(args) == 2:
        x, y = args

        if isinstance(x, Plane) and isinstance(y, Plane):
            l = x.meet(y)
            p = l.meet(infty_plane)
            polar = Line(p.array[:-1], copy=False)
            tangent_points = absolute_conic.intersect(polar)
            tangent_points = [
                Point(np.append(p.array, 0), copy=False) for p in tangent_points
            ]
            i = l.join(p.join(tangent_points[0]))
            j = l.join(p.join(tangent_points[1]))
            return 1 / 2j * np.log(crossratio(x, y, i, j))

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
            a = Point(basis.dot(a.array), copy=False)
            b = Point(basis.dot(x.meet(infty_plane).array), copy=False)
            c = Point(basis.dot(y.meet(infty_plane).array), copy=False)
        else:
            b = x.meet(infty)
            c = y.meet(infty)
    else:
        raise ValueError("Expected 2 or 3 arguments, got %s." % len(args))

    return np.real(1 / 2j * np.log(crossratio(b, c, I, J, a)))


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
        L = Point(basis.dot(l.meet(infty_plane).array), copy=False)
        M = Point(basis.dot(m.meet(infty_plane).array), copy=False)

    else:
        L, M = l.meet(infty), m.meet(infty)

    p = Point(0, 0)
    li = det([p.array, L.array, I.array])
    lj = det([p.array, L.array, J.array])
    mi = det([p.array, M.array, I.array])
    mj = det([p.array, M.array, J.array])
    a, b = np.sqrt(lj * mj), np.sqrt(li * mi)
    r, s = a * I + b * J, a * I - b * J

    if o.dim > 2:
        r, s = Point(basis.T.dot(r.array), copy=False), Point(
            basis.T.dot(s.array), copy=False
        )

    return Line(o, r), Line(o, s)


def dist(p, q):
    r"""Calculates the (euclidean) distance between two objects.

    Instead of the usual formula for the euclidean distance this function uses
    the following formula that is projectively invariant (in P and Q):

    .. math::
        \textrm{dist}(P, Q) = 4 \left|\frac{\sqrt{[P, Q, I][P, Q, J]}}{[P, I, J][Q, I, J]}\right|

    Parameters
    ----------
    p, q : Point, Line or Plane
        The points, lines or planes to calculate the distance between.

    Returns
    -------
    float
        The distance between the given objects.

    References
    ----------
    .. [1] J. Richter-Gebert: Perspectives on Projective Geometry, Section 18.8

    """
    if p == q:
        return 0

    if isinstance(p, (Plane, Line)) and isinstance(q, Point):
        return dist(p.project(q), q)
    if isinstance(p, Point) and isinstance(q, (Plane, Line)):
        return dist(q.project(p), p)
    if isinstance(p, Plane) and isinstance(q, (Plane, Line)):
        return dist(p, Point(q.basis_matrix[0, :], copy=False))
    if isinstance(q, Plane) and isinstance(p, Line):
        return dist(q, p.base_point)

    if p.dim > 2:
        x = np.stack([p.normalized_array, q.normalized_array], axis=-2)
        z = x[..., -1]

        m = orth(np.swapaxes(x, -1, -2), 2)
        x = np.matmul(x, m)
        x = np.concatenate([x, np.expand_dims(z, -1)], axis=-1)
        p, q, i, j = np.broadcast_arrays(x[..., 0, :], x[..., 1, :], I.array, J.array)
    else:
        p, q, i, j = np.broadcast_arrays(p.array, q.array, I.array, J.array)

    pqi = det(np.stack([p, q, i], axis=-2))
    pqj = det(np.stack([p, q, j], axis=-2))
    pij = det(np.stack([p, i, j], axis=-2))
    qij = det(np.stack([q, i, j], axis=-2))

    with np.errstate(divide="ignore", invalid="ignore"):
        return 4 * np.abs(np.sqrt(pqi * pqj) / (pij * qij))


def is_cocircular(a, b, c, d, rtol=EQ_TOL_REL, atol=EQ_TOL_ABS):
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
        a = Point(basis.dot(a.array), copy=False)
        b = Point(basis.dot(b.array), copy=False)
        c = Point(basis.dot(c.array), copy=False)
        d = Point(basis.dot(d.array), copy=False)

    i = crossratio(a, b, c, d, I)
    j = crossratio(a, b, c, d, J)
    return np.isclose(i, j, rtol, atol)


def is_perpendicular(l, m, rtol=EQ_TOL_REL, atol=EQ_TOL_ABS):
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
        L = Point(basis.dot(l.meet(infty_plane).array), copy=False)
        M = Point(basis.dot(m.meet(infty_plane).array), copy=False)

    elif isinstance(l, Plane) and isinstance(m, Plane):
        x = l.meet(m)
        p = x.meet(infty_plane)
        polar = Line(p.array[:-1], copy=False)
        tangent_points = absolute_conic.intersect(polar)
        tangent_points = [
            Point(np.append(p.array, 0), copy=False) for p in tangent_points
        ]
        i = x.join(p.join(tangent_points[0]))
        j = x.join(p.join(tangent_points[1]))
        return np.isclose(crossratio(l, m, i, j), -1, rtol, atol)

    else:
        raise NotImplementedError("Only two lines or two planes are supported.")

    return np.isclose(crossratio(L, M, I, J, Point(1, 1)), -1, rtol, atol)


def is_coplanar(*args, tol=EQ_TOL_ABS):
    """Tests whether the given points or lines are collinear, coplanar or concurrent. Works in any dimension.

    Due to line point duality this function has dual versions :obj:`is_collinear` and :obj:`is_concurrent`.

    Parameters
    ----------
    *args
        The points or lines to test.
    tol : float, optional
            The accepted tolerance.

    Returns
    -------
    array_like
        True if the given points are coplanar (in 3D) or collinear (in 2D) or if the given lines are concurrent.

    """
    n = args[0].dim + 1
    result = np.isclose(
        det(np.stack([a.array for a in args[:n]], axis=-2)), 0, atol=tol
    )
    if not np.any(result) or len(args) == n:
        return result
    covariant = args[0].tensor_shape[1] > 0
    e = LeviCivitaTensor(n, covariant=covariant)
    diagram = TensorDiagram(*[(e, a) if covariant else (a, e) for a in args[: n - 1]])
    tensor = diagram.calculate()
    for t in args[n:]:
        x = t * tensor if covariant else tensor * t
        result &= np.isclose(x.array, 0, atol=tol)
        if not np.any(result):
            break
    return result


is_collinear = is_coplanar
is_concurrent = is_coplanar
