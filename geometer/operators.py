from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt

from geometer.base import EQ_TOL_ABS, EQ_TOL_REL, LeviCivitaTensor, TensorDiagram
from geometer.curve import absolute_conic
from geometer.exceptions import NotCollinear, NotConcurrent
from geometer.point import (
    I,
    J,
    LineCollection,
    LineTensor,
    PlaneCollection,
    PlaneTensor,
    Point,
    PointCollection,
    PointTensor,
    SubspaceTensor,
    infty,
    infty_plane,
    join,
    meet,
)
from geometer.utils import det, matvec, orth

if TYPE_CHECKING:
    from geometer.shapes import PolytopeTensor


def crossratio(
    a: PointTensor | SubspaceTensor,
    b: PointTensor | SubspaceTensor,
    c: PointTensor | SubspaceTensor,
    d: PointTensor | SubspaceTensor,
    from_point: PointTensor | None = None,
) -> np.ndarray:
    """Calculates the cross ratio of points or lines.

    Args:
        a, b, c, d: The points, lines or planes (any dimension) to calculate the cross ratio of.
        from_point: A 2D point, only accepted if the other arguments are also 2D points.

    Returns:
        The cross ratio(s) of the given objects.

    Raises:
        NotConcurrent: If four lines are supplied that are not concurrent.
        NotCollinear: If four points are supplied that are not collinear.

    """
    if a == b:
        return np.ones(a.shape[: a.free_indices])

    if (
        isinstance(a, LineTensor)
        and isinstance(b, LineTensor)
        and isinstance(c, LineTensor)
        and isinstance(d, LineTensor)
    ):
        if not np.all(is_concurrent(a, b, c, d)):
            raise NotConcurrent("The lines are not concurrent: " + str([a, b, c, d]))

        from_point = a.meet(b)
        a, b, c, d = a.base_point, b.base_point, c.base_point, d.base_point

    elif (
        isinstance(a, PlaneTensor)
        and isinstance(b, PlaneTensor)
        and isinstance(c, PlaneTensor)
        and isinstance(d, PlaneTensor)
    ):
        l = a.meet(b)
        e = PlaneCollection.from_array(l.direction.array)
        a, b, c, d = e.meet(a), e.meet(b), e.meet(c), e.meet(d)
        m = e.basis_matrix
        p = e.meet(l)
        from_point = p._matrix_transform(m)
        a = (p + a.direction)._matrix_transform(m)
        b = (p + b.direction)._matrix_transform(m)
        c = (p + c.direction)._matrix_transform(m)
        d = (p + d.direction)._matrix_transform(m)

    elif not (
        isinstance(a, PointTensor)
        and isinstance(b, PointTensor)
        and isinstance(c, PointTensor)
        and isinstance(d, PointTensor)
    ):
        raise TypeError(f"Unsupported combination of types: a: {type(a)}, b: {type(b)}, c: {type(c)}, d: {type(d)}")

    if a.dim > 2 or (from_point is None and a.dim == 2):
        if not np.all(is_collinear(a, b, c, d)):
            raise NotCollinear("The points are not collinear: " + str([a, b, c, d]))

        basis = np.stack([a.array, b.array], axis=-2)
        a = matvec(basis, a.array)  # type: ignore[assignment]
        b = matvec(basis, b.array)  # type: ignore[assignment]
        c = matvec(basis, c.array)  # type: ignore[assignment]
        d = matvec(basis, d.array)  # type: ignore[assignment]
        o = []

    elif from_point is not None:
        a, b, c, d, from_point = np.broadcast_arrays(a.array, b.array, c.array, d.array, from_point.array)  # type: ignore[assignment]
        o = [from_point]
    else:
        a, b, c, d = np.broadcast_arrays(a.array, b.array, c.array, d.array)  # type: ignore[assignment]
        o = []

    ac = det(np.stack([*o, a, c], axis=-2))
    bd = det(np.stack([*o, b, d], axis=-2))
    ad = det(np.stack([*o, a, d], axis=-2))
    bc = det(np.stack([*o, b, c], axis=-2))

    with np.errstate(divide="ignore", invalid="ignore"):
        return ac * bd / (ad * bc)


def harmonic_set(a: PointTensor, b: PointTensor, c: PointTensor) -> PointTensor:
    """Constructs a fourth point that forms a harmonic set with the given points.

    The three given points must be collinear.

    If the returned point is d, the points {{a, b}, {c, d}} will be in harmonic position.

    Args:
        a, b, c: The points (any dimension) that are used to construct the fourth point in the harmonic set.

    Returns:
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
        o = cast(PointCollection, o._matrix_transform(basis))

        l = join(a, b)

    m = join(o, c)
    p = o + 1 / 2 * m.direction
    result = l.meet(join(meet(o.join(a), p.join(b)), meet(o.join(b), p.join(a))))  # type: ignore[call-arg, attr-defined]

    if n > 3:
        return result._matrix_transform(np.swapaxes(basis, -1, -2))

    return result


def angle(*args: PointTensor | LineTensor | PlaneTensor) -> npt.NDArray[np.float64]:
    r"""Calculates the (oriented) angle between given points, lines or planes.

    The function uses the Laguerre formula to calculate angles in two or three-dimensional projective space
    using cross ratios. To calculate the angle between two planes, two additional planes tangent to the absolute
    conic are constructed.

    Since the Laguerre formula uses the complex logarithm (which gives values between :math:`-\pi i` and :math:`\pi i`)
    and multiplies it with :math:`1/2i`, this function can only calculate angles between :math:`-\pi / 2` and
    :math:`\pi / 2`.

    The sign of the angle is determined by the order of the arguments. The points passed to the cross ratio are in
    the same order as the arguments to this function.
    When three points are given as arguments, the first point is the point at which the angle is calculated.

    Args:
        *args: The objects of which the function calculates the angle. These can be 2 or 3 points, 2 lines or 2 planes.

    Returns:
        The oriented angle(s) between the given objects.

    References:
      - Olivier Faugeras, Three-dimensional Computer Vision, Page 30

    """
    if len(args) == 3:
        a, b, c = args
        if a.dim > 2:
            e = join(*args)  # type: ignore[call-arg, arg-type, misc]
            basis = e.basis_matrix
            a = a._matrix_transform(basis)  # type: ignore[assignment]
            b = b._matrix_transform(basis)  # type: ignore[assignment]
            c = c._matrix_transform(basis)  # type: ignore[assignment]

    elif len(args) == 2:
        x, y = args

        if isinstance(x, PlaneTensor) and isinstance(y, PlaneTensor):
            l = x.meet(y)
            p = l.meet(infty_plane)
            polar = LineCollection.from_array(p.array[..., :-1])
            tangent_points = absolute_conic.intersect(polar)
            tangent_points = [
                PointCollection.from_array(np.append(p.array, np.zeros(p.shape[:-1] + (1,)), axis=-1))
                for p in tangent_points
            ]
            i = l.join(p.join(tangent_points[0]))
            j = l.join(p.join(tangent_points[1]))
            return 1 / 2j * np.log(crossratio(x, y, i, j))

        if isinstance(x, LineTensor) and isinstance(y, LineTensor):
            a = x.meet(y)
        else:
            a = Point(*(x.dim * [0]))
            if isinstance(x, PointTensor):
                x = a.join(x)  # type: ignore[assignment]
            if isinstance(y, PointTensor):
                y = a.join(y)  # type: ignore[assignment]

        if a.dim > 2:
            e = join(x, y)  # type: ignore[call-arg, arg-type, assignment]
            basis = e.basis_matrix
            a = a._matrix_transform(basis)
            b = x.meet(infty_plane)._matrix_transform(basis)  # type: ignore[union-attr]
            c = y.meet(infty_plane)._matrix_transform(basis)  # type: ignore[union-attr]
        else:
            b = x.meet(infty)  # type: ignore[union-attr]
            c = y.meet(infty)  # type: ignore[union-attr]
    else:
        raise ValueError(f"Expected 2 or 3 arguments, got {len(args)}.")

    return np.real(1 / 2j * np.log(crossratio(b, c, I, J, a)))  # type: ignore[arg-type]


def angle_bisectors(l: LineTensor, m: LineTensor) -> tuple[LineTensor, LineTensor]:
    """Constructs the angle bisectors of two given lines.

    Args:
        l, m: Two lines in any dimension.

    Returns:
        The two angle bisectors.

    """
    o = l.meet(m)

    if o.dim > 2:
        e = join(l, m)
        basis = e.basis_matrix
        L = l.meet(infty_plane)._matrix_transform(basis)
        M = m.meet(infty_plane)._matrix_transform(basis)

    else:
        L, M = l.meet(infty), m.meet(infty)  # type: ignore[assignment]

    p, L, M, i, j = np.broadcast_arrays([0, 0, 1], L.array, M.array, I.array, J.array)  # type: ignore[assignment]
    li = det(np.stack([p, L, i], axis=-2))
    lj = det(np.stack([p, L, j], axis=-2))
    mi = det(np.stack([p, M, i], axis=-2))
    mj = det(np.stack([p, M, j], axis=-2))
    a, b = np.sqrt(lj * mj), np.sqrt(li * mi)
    a, b = np.expand_dims(a, -1), np.expand_dims(b, -1)
    r, s = a * i + b * j, a * i - b * j

    r = PointCollection.from_array(r)
    s = PointCollection.from_array(s)

    if o.dim > 2:
        r = r._matrix_transform(np.swapaxes(basis, -1, -2))
        s = s._matrix_transform(np.swapaxes(basis, -1, -2))

    return join(o, r), join(o, s)


def _point_dist(p: PointTensor, q: PointTensor) -> npt.NDArray[np.float64]:
    if p.dim > 2:
        x = np.stack(np.broadcast_arrays(p.normalized_array, q.normalized_array), axis=-2)
        z = x[..., -1]

        m = orth(np.swapaxes(x, -1, -2), 2)
        x = np.matmul(x, m)
        x = np.concatenate([x, np.expand_dims(z, -1)], axis=-1)
        p, q, i, j = np.broadcast_arrays(x[..., 0, :], x[..., 1, :], I.array, J.array)  # type: ignore[assignment]
    else:
        p, q, i, j = np.broadcast_arrays(p.array, q.array, I.array, J.array)  # type: ignore[assignment]

    pqi = det(np.stack([p, q, i], axis=-2))
    pqj = det(np.stack([p, q, j], axis=-2))
    pij = det(np.stack([p, i, j], axis=-2))
    qij = det(np.stack([q, i, j], axis=-2))

    with np.errstate(divide="ignore", invalid="ignore"):
        return 4 * np.abs(np.sqrt(pqi * pqj) / (pij * qij))


def dist(
    p: PointTensor | SubspaceTensor | PolytopeTensor, q: PointTensor | SubspaceTensor | PolytopeTensor
) -> npt.NDArray[np.float64]:
    r"""Calculates the (euclidean) distance between two objects.

    Instead of the usual formula for the euclidean distance this function uses
    the following formula that is projectively invariant (in P and Q):

    .. math::
        \textrm{dist}(P, Q) = 4 \left|\frac{\sqrt{[P, Q, I][P, Q, J]}}{[P, I, J][Q, I, J]}\right|

    Args:
        p, q: The points, lines or planes to calculate the distance between.

    Returns:
        The distance between the given objects.

    References:
      - J. Richter-Gebert: Perspectives on Projective Geometry, Section 18.8

    """
    if p == q:
        return np.zeros(p.shape[: p.free_indices])

    if isinstance(p, PointTensor) and isinstance(q, PointTensor):
        return _point_dist(p, q)
    if isinstance(p, SubspaceTensor) and isinstance(q, PointTensor):
        return dist(p.project(q), q)
    if isinstance(p, PointTensor) and isinstance(q, SubspaceTensor):
        return dist(q.project(p), p)
    if isinstance(p, SubspaceTensor) and isinstance(q, PlaneTensor):
        return dist(q, p)
    if isinstance(p, PlaneTensor) and isinstance(q, LineTensor):
        return dist(p, q.base_point)
    if isinstance(p, PlaneTensor) and isinstance(q, SubspaceTensor):
        return dist(p, cast(PointCollection, PointCollection.from_array(q.basis_matrix[0, :])))

    from geometer.shapes import PolygonTensor, Polyhedron, SegmentTensor

    if isinstance(p, PointTensor) and isinstance(q, SegmentTensor):
        return dist(q, p)
    if isinstance(p, SegmentTensor) and isinstance(q, PointTensor):
        result = np.min([dist(v, q) for v in p.vertices], axis=0)
        r = p._line.project(q)
        return np.where(p.contains(r), dist(r, q), result)
    if isinstance(p, PointTensor) and isinstance(q, PolygonTensor):
        return dist(q, p)
    if isinstance(p, PolygonTensor) and isinstance(q, PointTensor):
        result = np.min([dist(e, q) for e in p.edges], axis=0)
        if p.dim > 2:
            r = p._plane.project(q)
            return np.where(p.contains(r), dist(r, q), result)
        return result
    if isinstance(p, PointTensor) and isinstance(q, Polyhedron):
        return dist(q, p)
    if isinstance(p, Polyhedron) and isinstance(q, PointTensor):
        return np.min([dist(f, q) for f in p.faces], axis=0)

    raise TypeError(f"Unsupported types {type(p)} and {type(q)}.")


def is_cocircular(
    a: PointTensor, b: PointTensor, c: PointTensor, d: PointTensor, rtol: float = EQ_TOL_REL, atol: float = EQ_TOL_ABS
) -> npt.NDArray[np.bool_]:
    """Tests whether four points lie on a circle.

    Args:
        a, b, c, d: Four points in RP2 or CP1.
        rtol: The relative tolerance parameter.
        atol: The absolute tolerance parameter.

    Returns:
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


def is_perpendicular(
    l: LineTensor | PlaneTensor, m: LineTensor | PlaneTensor, rtol: float = EQ_TOL_REL, atol: float = EQ_TOL_ABS
) -> npt.NDArray[np.bool_]:
    """Tests whether two lines/planes are perpendicular.

    Args:
        l, m: Two lines in any dimension or two planes in 3D.
        rtol: The relative tolerance parameter.
        atol: The absolute tolerance parameter.

    Returns:
        True if the two lines/planes are perpendicular.

    """
    if l.dim < 3:
        L = l.meet(infty)
        M = m.meet(infty)

    elif isinstance(l, LineTensor) and isinstance(m, LineTensor):
        e = join(l, m)
        basis = e.basis_matrix
        L = l.meet(infty_plane)._matrix_transform(basis)
        M = m.meet(infty_plane)._matrix_transform(basis)

    elif isinstance(l, PlaneTensor) and isinstance(m, PlaneTensor):
        x = l.meet(m)
        p = x.meet(infty_plane)
        polar = LineCollection.from_array(p.array[..., :-1])
        tangent_points = absolute_conic.intersect(polar)
        tangent_points = [
            PointCollection.from_array(np.append(p.array, np.zeros(p.shape[:-1] + (1,)), axis=-1))
            for p in tangent_points
        ]
        i = x.join(p.join(tangent_points[0]))
        j = x.join(p.join(tangent_points[1]))
        return np.isclose(crossratio(l, m, i, j), -1, rtol, atol)

    else:
        raise NotImplementedError("Only two lines or two planes are supported.")

    return np.isclose(crossratio(L, M, I, J, Point(1, 1)), -1, rtol, atol)


def is_coplanar(*args: PointTensor | LineTensor, tol: float = EQ_TOL_ABS) -> npt.NDArray[np.bool_]:
    """Tests whether the given points or lines are collinear, coplanar or concurrent. Works in any dimension.

    Due to line point duality this function has dual versions :obj:`is_collinear` and :obj:`is_concurrent`.

    Args:
        *args: The points or lines to test.
        tol: The accepted tolerance.

    Returns:
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
