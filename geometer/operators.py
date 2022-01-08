import numpy as np

from .base import EQ_TOL_ABS, EQ_TOL_REL, LeviCivitaTensor, TensorDiagram
from .curve import absolute_conic
from .exceptions import IncidenceError, NotCollinear
from .point import (I, J, Line, LineCollection, Plane, PlaneCollection, Point, PointCollection, infty, infty_plane,
                    join, meet)
from .utils import det, matvec, orth


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
    array_like
        The cross ratio(s) of the given objects.

    """

    if a == b:
        return np.ones(a.shape[: len(a._collection_indices)])

    if isinstance(a, (Line, LineCollection)):
        if not np.all(is_concurrent(a, b, c, d)):
            raise IncidenceError("The lines are not concurrent: " + str([a, b, c, d]))

        from_point = a.meet(b)
        a, b, c, d = a.base_point, b.base_point, c.base_point, d.base_point

    if isinstance(a, (Plane, PlaneCollection)):
        l = a.meet(b)
        if isinstance(l, Line):
            e = Plane(l.direction.array, copy=False)
        else:
            e = PlaneCollection(l.direction.array, copy=False)
        a, b, c, d = e.meet(a), e.meet(b), e.meet(c), e.meet(d)
        m = e.basis_matrix
        p = e.meet(l)
        from_point = p._matrix_transform(m)
        a = (p + a.direction)._matrix_transform(m)
        b = (p + b.direction)._matrix_transform(m)
        c = (p + c.direction)._matrix_transform(m)
        d = (p + d.direction)._matrix_transform(m)

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
        a, b, c, d, from_point = np.broadcast_arrays(a.array, b.array, c.array, d.array, from_point.array)
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
        e = join(l, o, _normalize_result=False)
        basis = e.basis_matrix
        a = a._matrix_transform(basis)
        b = b._matrix_transform(basis)
        c = c._matrix_transform(basis)
        o = o._matrix_transform(basis)

        l = join(a, b)

    m = join(o, c)
    p = o + 1 / 2 * m.direction
    result = l.meet(join(meet(o.join(a), p.join(b)), meet(o.join(b), p.join(a))))

    if n > 3:
        return result._matrix_transform(np.swapaxes(basis, -1, -2))

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
    array_like
        The oriented angle(s) between the given objects.

    References
    ----------
    .. [1] Olivier Faugeras, Three-dimensional Computer Vision, Page 30

    """
    if len(args) == 3:
        a, b, c = args
        if a.dim > 2:
            e = join(*args)
            basis = e.basis_matrix
            a = a._matrix_transform(basis)
            b = b._matrix_transform(basis)
            c = c._matrix_transform(basis)

    elif len(args) == 2:
        x, y = args

        if isinstance(x, Plane) and isinstance(y, Plane):
            l = x.meet(y)
            p = l.meet(infty_plane)
            polar = Line(p.array[:-1], copy=False)
            tangent_points = absolute_conic.intersect(polar)
            tangent_points = [Point(np.append(p.array, 0), copy=False) for p in tangent_points]
            i = l.join(p.join(tangent_points[0]))
            j = l.join(p.join(tangent_points[1]))
            return 1 / 2j * np.log(crossratio(x, y, i, j))

        if isinstance(x, PlaneCollection) and isinstance(y, PlaneCollection):
            l = x.meet(y)
            p = l.meet(infty_plane)
            polar = LineCollection(p.array[..., :-1], copy=False)
            tangent_points = absolute_conic.intersect(polar)
            tangent_points = [
                PointCollection(np.append(p.array, np.zeros(p.shape[:-1] + (1,)), axis=-1), copy=False)
                for p in tangent_points
            ]
            i = l.join(p.join(tangent_points[0]))
            j = l.join(p.join(tangent_points[1]))
            return 1 / 2j * np.log(crossratio(x, y, i, j))

        if isinstance(x, (Line, LineCollection)) and isinstance(
            y, (Line, LineCollection)
        ):
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
            a = a._matrix_transform(basis)
            b = x.meet(infty_plane)._matrix_transform(basis)
            c = y.meet(infty_plane)._matrix_transform(basis)
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
    l, m : Line, LineCollection
        Two lines in any dimension.

    Returns
    -------
    tuple of Line or tuple of LineCollection
        The two angle bisectors.

    """
    o = l.meet(m)

    if o.dim > 2:
        e = join(l, m)
        basis = e.basis_matrix
        L = l.meet(infty_plane)._matrix_transform(basis)
        M = m.meet(infty_plane)._matrix_transform(basis)

    else:
        L, M = l.meet(infty), m.meet(infty)

    p, L, M, i, j = np.broadcast_arrays([0, 0, 1], L.array, M.array, I.array, J.array)
    li = det(np.stack([p, L, i], axis=-2))
    lj = det(np.stack([p, L, j], axis=-2))
    mi = det(np.stack([p, M, i], axis=-2))
    mj = det(np.stack([p, M, j], axis=-2))
    a, b = np.sqrt(lj * mj), np.sqrt(li * mi)
    a, b = np.expand_dims(a, -1), np.expand_dims(b, -1)
    r, s = a * i + b * j, a * i - b * j

    if r.ndim > 1:
        r = PointCollection(r, copy=False)
        s = PointCollection(s, copy=False)
    else:
        r = Point(r, copy=False)
        s = Point(s, copy=False)

    if o.dim > 2:
        r = r._matrix_transform(np.swapaxes(basis, -1, -2))
        s = s._matrix_transform(np.swapaxes(basis, -1, -2))

    return join(o, r), join(o, s)


def dist(p, q):
    r"""Calculates the (euclidean) distance between two objects.

    Instead of the usual formula for the euclidean distance this function uses
    the following formula that is projectively invariant (in P and Q):

    .. math::
        \textrm{dist}(P, Q) = 4 \left|\frac{\sqrt{[P, Q, I][P, Q, J]}}{[P, I, J][Q, I, J]}\right|

    Parameters
    ----------
    p, q : Point, Line, Plane or Polygon
        The points, lines or planes to calculate the distance between.

    Returns
    -------
    array_like
        The distance between the given objects.

    References
    ----------
    .. [1] J. Richter-Gebert: Perspectives on Projective Geometry, Section 18.8

    """
    if p == q:
        return 0

    point_types = (Point, PointCollection)
    line_types = (Line, LineCollection)
    plane_types = (Plane, PlaneCollection)
    subspace_types = plane_types + line_types

    if isinstance(p, subspace_types) and isinstance(q, point_types):
        return dist(p.project(q), q)
    if isinstance(p, point_types) and isinstance(q, subspace_types):
        return dist(q.project(p), p)
    if isinstance(p, plane_types) and isinstance(q, subspace_types):
        return dist(p, Point(q.basis_matrix[0, :], copy=False))
    if isinstance(q, plane_types) and isinstance(p, line_types):
        return dist(q, p.base_point)

    from .shapes import Polygon, Polyhedron, Segment

    if isinstance(p, point_types) and isinstance(q, Polygon):
        return dist(q, p)
    if isinstance(p, Polygon) and isinstance(q, point_types):
        result = np.min([dist(e, q) for e in p.edges], axis=0)
        if p.dim > 2:
            r = p._plane.project(q)
            return np.where(p.contains(r), dist(r, q), result)
        return result
    if isinstance(p, point_types) and isinstance(q, Polyhedron):
        return dist(q, p)
    if isinstance(p, Polyhedron) and isinstance(q, point_types):
        return np.min([dist(f, q) for f in p.faces], axis=0)
    if isinstance(p, point_types) and isinstance(q, Segment):
        return dist(q, p)
    if isinstance(p, Segment) and isinstance(q, point_types):
        result = np.min([dist(v, q) for v in p.vertices], axis=0)
        r = p._line.project(q)
        return np.where(p.contains(r), dist(r, q), result)

    if not isinstance(p, point_types) or not isinstance(q, point_types):
        raise TypeError("Unsupported types %s and %s." % (type(p), type(q)))

    if p.dim > 2:
        x = np.stack(np.broadcast_arrays(p.normalized_array, q.normalized_array), axis=-2)
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
    array_like
        True if the four points lie on a circle.

    """
    if a.dim == 1:
        return np.isreal(crossratio(a, b, c, d))

    elif a.dim > 2:
        e = join(a, b, c)
        basis = e.basis_matrix
        a = a._matrix_transform(basis)
        b = b._matrix_transform(basis)
        c = c._matrix_transform(basis)
        d = d._matrix_transform(basis)

    i = crossratio(a, b, c, d, I)
    j = crossratio(a, b, c, d, J)
    return np.isclose(i, j, rtol, atol)


def is_perpendicular(l, m, rtol=EQ_TOL_REL, atol=EQ_TOL_ABS):
    """Tests whether two lines/planes are perpendicular.

    Parameters
    ----------
    l, m : Line, LineCollection, Plane or PlaneCollection
        Two lines in any dimension or two planes in 3D.
    rtol : float, optional
        The relative tolerance parameter.
    atol : float, optional
        The absolute tolerance parameter.

    Returns
    -------
    array_like
        True if the two lines/planes are perpendicular.

    """
    if l.dim < 3:
        L = l.meet(infty)
        M = m.meet(infty)

    elif isinstance(l, (Line, LineCollection)) and isinstance(m, (Line, LineCollection)):
        e = join(l, m)
        basis = e.basis_matrix
        L = l.meet(infty_plane)._matrix_transform(basis)
        M = m.meet(infty_plane)._matrix_transform(basis)

    elif isinstance(l, Plane) and isinstance(m, Plane):
        x = l.meet(m)
        p = x.meet(infty_plane)
        polar = Line(p.array[:-1], copy=False)
        tangent_points = absolute_conic.intersect(polar)
        tangent_points = [Point(np.append(p.array, 0), copy=False) for p in tangent_points]
        i = x.join(p.join(tangent_points[0]))
        j = x.join(p.join(tangent_points[1]))
        return np.isclose(crossratio(l, m, i, j), -1, rtol, atol)

    elif isinstance(l, PlaneCollection) and isinstance(m, PlaneCollection):
        x = l.meet(m)
        p = x.meet(infty_plane)
        polar = LineCollection(p.array[..., :-1], copy=False)
        tangent_points = absolute_conic.intersect(polar)
        tangent_points = [
            PointCollection(np.append(p.array, np.zeros(p.shape[:-1] + (1,)), axis=-1), copy=False)
            for p in tangent_points
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
    result = np.isclose(det(np.stack([a.array for a in args[:n]], axis=-2)), 0, atol=tol)
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
