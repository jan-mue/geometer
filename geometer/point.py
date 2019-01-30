from collections import Iterable

import numpy as np
import sympy
import scipy.linalg
from .base import ProjectiveElement, TensorDiagram, LeviCivitaTensor, Tensor
from .exceptions import LinearDependenceError


def _join_meet_duality(*args, intersect_lines=True):
    if len(args) < 2:
        raise ValueError("Expected at least 2 arguments, got %s." % str(len(args)))

    n = args[0].dim + 1

    if all(o.tensor_shape == args[0].tensor_shape for o in args[1:]) and sum(args[0].tensor_shape) == 1:
        covariant = args[0].tensor_shape[0] > 0
        e = LeviCivitaTensor(n, not covariant)
        diagram = TensorDiagram(*[(o, e) if covariant else (e, o) for o in args])
        result = diagram.calculate()

    elif len(args) == 2:
        a, b = args
        if isinstance(a, Line) and isinstance(b, Plane) or isinstance(b, Line) and isinstance(a, Plane):
            e = LeviCivitaTensor(n)
            d = TensorDiagram(*[(e, a)] * a.tensor_shape[1], *[(e, b)] * b.tensor_shape[1])
            result = d.calculate()
        elif isinstance(a, Line) and isinstance(b, Point):
            result = a * b
        elif isinstance(a, Point) and isinstance(b, Line):
            result = b * a
        elif isinstance(a, Line) and isinstance(b, Line):
            l, m = args
            a = (m * l.covariant_tensor).array
            i = np.unravel_index(np.abs(a).argmax(), a.shape)
            if not intersect_lines:
                result = Tensor(a[i[0], :], covariant=False)
            else:
                result = Tensor(a[:, i[1]])
        else:
            raise ValueError("Operation not supported.")

    else:
        raise ValueError("Wrong number of arguments.")

    if result == 0:
        raise LinearDependenceError("Arguments are not linearly independent.")

    if result.tensor_shape == (0, 1):
        return Line(result) if n == 3 else Plane(result)
    if result.tensor_shape == (1, 0):
        return Point(result)
    if result.tensor_shape == (2, 0):
        return Line(result).contravariant_tensor
    if result.tensor_shape[0] == 0:
        return Line(result)

    raise ValueError("Unexpected tensor shape: " + str(result.tensor_shape))


def join(*args):
    """Joins a number of objects to form a line or plane.

    Parameters
    ----------
    *args
        Objects to join, e.g. 2 points, lines, a point and a line or 3 points.

    Returns
    -------
    :obj:`Line` or :obj:`Plane`
        The resulting plane or line.

    """
    return _join_meet_duality(*args, intersect_lines=False)


def meet(*args):
    """Intersects a number of given objects.

    Parameters
    ----------
    *args
        Objects to intersect, e.g. two lines, planes, a plane and a line or 3 planes.

    Returns
    -------
    :obj:`Point` or :obj:`Line`
        The resulting point or line.

    """
    return _join_meet_duality(*args, intersect_lines=True)


class Point(ProjectiveElement):
    """Represents points in a projective space of arbitrary dimension.

    The number of supplied coordinates determines the dimension of the space that the vector lives in.
    If the coordinates are given as arguments (not in a single iterable), the coordinates will automatically be
    transformed into homogeneous coordinates, i.e. a one added as an additional coordinate.

    Parameters
    ----------
    *args
        A single iterable object or tensor or multiple (affine) coordinates.

    """

    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], (Iterable, Tensor)):
            super(Point, self).__init__(*args)
        else:
            super(Point, self).__init__(*args, 1)

    def __add__(self, other):
        a, b = self.normalized_array, other.normalized_array
        result = a[:-1] + b[:-1]
        result = np.append(result, min(a[-1], b[-1]))
        return Point(result)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if not np.isscalar(other):
            return super(Point, self).__mul__(other)
        result = self.array[:-1] * other
        result = np.append(result, self.array[-1])
        return Point(result)

    def __rmul__(self, other):
        if not np.isscalar(other):
            return super(Point, self).__rmul__(other)
        return self * other

    def __neg__(self):
        return (-1) * self

    def __repr__(self):
        return "Point({})".format(",".join(self.normalized_array[:-1].astype(str))) + (" at Infinity" if np.isclose(self.array[-1], 0) else "")

    @property
    def normalized_array(self):
        """numpy.ndarray: The normalized coordinates as array."""
        if np.isclose(self.array[-1], 0):
            return np.real_if_close(self.array)
        return np.real_if_close(self.array / self.array[-1])

    def join(self, *others):
        """Execute the join of this point with other objects.

        Parameters
        ----------
        *others
            The objects to join the point with.

        Returns
        -------
        :obj:`Line` or :obj:`Plane`
            The result of the join operation.

        See Also
        --------
        join

        """
        return join(self, *others)


I = Point([-1j, 1, 0])
J = Point([1j, 1, 0])


class Line(ProjectiveElement):
    """Represents a line in a projective space of arbitrary dimension.

    Parameters
    ----------
    *args
        Two points or the coordinates of the line. Instead of all coordinates separately, a single iterable can also
        be supplied.

    """

    def __init__(self, *args):
        if len(args) == 2:
            pt1, pt2 = args
            super(Line, self).__init__(pt1.join(pt2))
        else:
            super(Line, self).__init__(*args, covariant=False)

    def polynomials(self, symbols=None):
        """Returns a list of polynomials, to use for symbolic calculations.

        Parameters
        ----------
        symbols : :obj:`list` of :obj:`sympy.Symbol`, optional
            The symbols used in the resulting polynomial. By default "x1", ..., "xn" will be used.

        Returns
        -------

        """
        symbols = symbols or sympy.symbols(["x" + str(i) for i in range(self.array.shape[0])])

        def p(row):
            f = sum(x * s for x, s in zip(row, symbols))
            return sympy.poly(f, symbols)

        if self.dim == 2:
            return [p(self.array)]
        return np.apply_along_axis(p, axis=1, arr=self.array[np.any(self.array, axis=1)])

    @property
    def covariant_tensor(self):
        """Line: The covariant version of a line in 3D."""
        if self.tensor_shape[0] > 0:
            return self
        e = LeviCivitaTensor(4)
        diagram = TensorDiagram((e, self), (e, self))
        return Line(diagram.calculate())

    @property
    def contravariant_tensor(self):
        """Line: The contravariant version of a line in 3D."""
        if self.tensor_shape[1] > 0:
            return self
        e = LeviCivitaTensor(4, False)
        diagram = TensorDiagram((self, e), (self, e))
        return Line(diagram.calculate())

    def contains(self, pt: Point):
        """Tests if a given point lies on the line.

        Parameters
        ----------
        pt: Point
            The point to test.

        Returns
        -------
        bool
            True if the line contains the point.

        """
        return self * pt == 0

    def meet(self, *others):
        """Intersect the line with other objects.

        Parameters
        ----------
        *others
            The objects to intersect the line with.

        Returns
        -------
        Point
            The result of the meet operation.

        See Also
        --------
        meet

        """
        return meet(self, *others)

    def join(self, *others):
        """Execute the join of line point with other objects.

        Parameters
        ----------
        *others
            The objects to join the line with.

        Returns
        -------
        Plane
            The result of the join operation.

        See Also
        --------
        join

        """
        return join(self, *others)

    def is_coplanar(self, other):
        """Tests whether another line lies in the same plane as this line, i.e. whether two lines intersect in 3D.

        Parameters
        ----------
        other : Line
            A line in 3D to test.

        Returns
        -------
        bool
            True if the two lines intersect (i.e. they lie in the same plane).

        """
        l = other.covariant_tensor
        d = TensorDiagram((l, self), (l, self))
        return d.calculate() == 0

    def __add__(self, point):
        t = np.array([[1, 0, 0], [0, 1, 0], (-point.normalized_array)]).T
        return Line(self.array.dot(t))

    def __radd__(self, other):
        return self + other

    def __repr__(self):
        return "Line({})".format(str(self.array.tolist()))

    def parallel(self, through):
        p = self.meet(infty_hyperplane(self.dim))
        return p.join(through)

    def is_parallel(self, other):
        p = self.meet(other)
        return np.isclose(p.array[-1], 0)

    def perpendicular(self, through):
        return self.mirror(through).join(through)

    def project(self, pt: Point):
        l = self.mirror(pt).join(pt)
        return self.meet(l)

    @property
    def base_point(self):
        if self.dim > 2:
            return Point(self.basis_matrix[0, :])

        if np.isclose(self.array[2], 0):
            return Point(0, 0)

        if not np.isclose(self.array[1], 0):
            return Point([0, -self.array[2], self.array[1]])

        return Point([self.array[2], 0, -self.array[0]])

    @property
    def direction(self):
        if self.dim > 2:
            base = self.basis_matrix
            return Point(base[0, :] - base[1, :])
        if np.isclose(self.array[0], 0) and np.isclose(self.array[1], 0):
            return Point([0, 1, 0])
        return Point(self.array[1], -self.array[0])

    @property
    def basis_matrix(self):
        if self.dim == 2:
            a = self.base_point.array
            b = np.cross(self.array, a)
            result = np.array([a, b])
            return result / np.linalg.norm(result)
        return scipy.linalg.null_space(self.array).T

    def mirror(self, pt):
        l = self
        if self.dim == 3:
            e = Plane(self, pt)
            m = e.basis_matrix
            pt = Point(m.dot(pt.array))
            basis = scipy.linalg.null_space(self.array)
            a, b = m.dot(basis).T
            l = Line(Point(a), Point(b))
        l1 = I.join(pt)
        l2 = J.join(pt)
        p1 = l.meet(l1)
        p2 = l.meet(l2)
        m1 = p1.join(J)
        m2 = p2.join(I)
        result = m1.meet(m2)
        if self.dim == 3:
            return Point(m.T.dot(result.array))
        return result


infty = Line(0, 0, 1)


class Plane(ProjectiveElement):

    def __init__(self, *args):
        if all(isinstance(o, (Line, Point)) for o in args):
            super(Plane, self).__init__(join(*args))
        else:
            super(Plane, self).__init__(*args, covariant=False)

    def contains(self, other):
        if isinstance(other, Point):
            return np.isclose(np.vdot(self.array, other.array), 0)
        elif isinstance(other, Line):
            return self * other.covariant_tensor == 0

    def meet(self, *others):
        """Intersect the line with other objects.

        Parameters
        ----------
        *others
            The objects to intersect the plane with.

        Returns
        -------
        :obj:`Line` or :obj:`Point`
            The result of the meet operation.

        See Also
        --------
        meet

        """
        return meet(self, *others)

    @property
    def basis_matrix(self):
        n = self.dim + 1
        i = np.where(self.array != 0)[0][0]
        result = np.zeros((n, n - 1), dtype=self.array.dtype)
        a = [j for j in range(n) if j != i]
        result[i, :] = self.array[a]
        result[a, range(n - 1)] = -self.array[i]
        q, r = np.linalg.qr(result)
        return q.T

    @property
    def polynomial(self):
        symbols = sympy.symbols(["x" + str(i) for i in range(self.dim + 1)])
        f = sum(x * s for x, s in zip(self.array, symbols))
        return sympy.poly(f, symbols)

    def __repr__(self):
        return "Plane({})".format(",".join(self.array.astype(str)))

    def parallel(self, through):
        l = self.meet(infty_hyperplane(self.dim))
        return join(l, through)

    def mirror(self, pt):
        l = self.meet(infty_plane)
        l = Line(np.cross(*l.basis_matrix[:, :-1]))
        p = l.base_point
        polar = Line(p.array)

        from .curve import absolute_conic
        tangent_points = absolute_conic.intersect(polar)
        tangent_points = [Point(np.append(p.array, 0)) for p in tangent_points]

        l1 = tangent_points[0].join(pt)
        l2 = tangent_points[1].join(pt)
        p1 = self.meet(l1)
        p2 = self.meet(l2)
        m1 = p1.join(tangent_points[1])
        m2 = p2.join(tangent_points[0])
        return m1.meet(m2)

    def project(self, pt):
        l = self.mirror(pt).join(pt)
        return self.meet(l)

    def perpendicular(self, through: Point):
        return self.mirror(through).join(through)


def infty_hyperplane(dimension):
    if dimension == 2:
        return infty
    return Plane([0] * dimension + [1])


infty_plane = infty_hyperplane(3)
