from collections import Iterable

import numpy as np
import sympy

from .base import ProjectiveElement, TensorDiagram, LeviCivitaTensor, Tensor, _symbols
from .utils import null_space
from .exceptions import LinearDependenceError, NotCoplanar


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
            e = LeviCivitaTensor(n)

            if n > 4:
                diagram = TensorDiagram(*[(e, a)] * a.tensor_shape[1], *[(e, b)] * (n-a.tensor_shape[1]))
                result = diagram.calculate()
                coplanar = result == 0
            else:
                coplanar = n < 4 or a.is_coplanar(b)

            if not coplanar and intersect_lines:
                raise NotCoplanar("The given lines are not coplanar.")

            if coplanar:
                diagram = TensorDiagram(*[(e, a)] * a.tensor_shape[1], (e, b))
                array = diagram.calculate().array
                i = np.unravel_index(np.abs(array).argmax(), array.shape)
                if not intersect_lines:
                    result = Tensor(array[i[0], ...], covariant=False)
                else:
                    result = Tensor(array[(slice(None),) + i[1:]])
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
    if result.tensor_shape == (0, n-2):
        return Line(result)

    return Subspace(result)


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
        return "Point({})".format(", ".join(self.normalized_array[:-1].astype(str))) + (" at Infinity" if np.isclose(self.array[-1], 0) else "")

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


class Subspace(ProjectiveElement):
    """Represents a general subspace of a projective space. Line and Plane are subclasses.

    Parameters
    ----------
    *args
        The coordinates of the subspace. Instead of all coordinates separately, a single iterable can also be supplied.

    """

    def __init__(self, *args):
        super(Subspace, self).__init__(*args, covariant=False)

    def polynomials(self, symbols=None):
        """Returns a list of polynomials, to use for symbolic calculations.

        Parameters
        ----------
        symbols : :obj:`list` of :obj:`sympy.Symbol`, optional
            The symbols used in the resulting polynomial. By default "x1", ..., "xn" will be used.

        Returns
        -------
        list of sympy.Poly
            The polynomials describing the subspace.

        """
        symbols = symbols or _symbols(self.array.shape[0])

        def p(row):
            f = sum(x * s for x, s in zip(row, symbols))
            return sympy.poly(f, symbols)

        if self.dim == 2:
            return [p(self.array)]
        return np.apply_along_axis(p, axis=1, arr=self.array[np.any(self.array, axis=1)])

    @property
    def basis_matrix(self):
        """numpy.ndarray: A matrix with orthonormal basis vectors as rows."""
        x = self.array
        if len(x.shape) > 2:
            x = self.array.reshape((x.shape[0]**(len(x.shape)-1), x.shape[-1]))
        return null_space(x).T

    def contains(self, other):
        """Tests whether a given point or line lies in the subspace.

        Parameters
        ----------
        other : :obj:`Point` or :obj:`Line`
            The object to test.

        Returns
        -------
        bool
            True, if the given point/line lies in the subspace.

        """
        if isinstance(other, Point):
            return self * other == 0
        elif isinstance(other, Line):
            return self * other.covariant_tensor == 0

    def meet(self, *others):
        """Intersect the subspace with other objects.

        Parameters
        ----------
        *others
            The objects to intersect the subspace with.

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
        """Execute the join of the subspace with other objects.

        Parameters
        ----------
        *others
            The objects to join the subspace with.

        Returns
        -------
        Plane
            The result of the join operation.

        See Also
        --------
        join

        """
        return join(self, *others)

    def parallel(self, through):
        """Returns the subspace through a given point that is parallel to this subspace.

        Parameters
        ----------
        through : Point
            The point through which the parallel subspace is to be constructed.

        Returns
        -------
        Subspace
            The parallel subspace.

        """
        x = self.meet(infty_hyperplane(self.dim))
        return join(x, through)

    def is_parallel(self, other):
        """Tests whether a given subspace is parallel to this subspace.

        Parameters
        ----------
        other : Subspace
            The other space to test.

        Returns
        -------
        bool
            True, if the two spaces are parallel.

        """
        x = self.meet(other)
        return infty_hyperplane(self.dim).contains(x)


class Line(Subspace):
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
            super(Line, self).__init__(*args)

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

    def is_coplanar(self, other):
        """Tests whether another line lies in the same plane as this line, i.e. whether two lines intersect.

        Parameters
        ----------
        other : Line
            A line in 3D to test.

        Returns
        -------
        bool
            True if the two lines intersect (i.e. they lie in the same plane).

        """
        if self.dim == 2:
            return True

        e = LeviCivitaTensor(self.dim + 1)
        diagram = TensorDiagram(*[(e, self)]*(self.dim - 1), *[(e, other)]*(self.dim - 1))
        return diagram.calculate() == 0

    def __add__(self, point):
        from .transformation import translation
        return translation(*point.normalized_array[:-1])*self

    def __radd__(self, other):
        return self + other

    def perpendicular(self, through):
        """Construct the perpendicular line though a point.

        Parameters
        ----------
        through : Point
            The point through which the perpendicular is constructed.

        Returns
        -------
        Line
            The perpendicular line.

        """
        if self.contains(through):
            n = self.dim + 1

            l = self

            if n > 3:
                # additional point is required to determine the exact line
                arr = np.zeros(n)
                for i in range(n):
                    arr[-i - 1] = 1
                    o = Point(arr)
                    if not self.contains(o):
                        break
                e = join(self, o)
                basis = e.basis_matrix
                line_pts = basis.dot(self.basis_matrix.T)
                l = Line(np.cross(*line_pts.T))

            from .operators import harmonic_set
            p = l.meet(infty)
            q = harmonic_set(I, J, p)

            if n > 3:
                q = Point(basis.T.dot(q.array))

            return Line(through, q)

        return self.mirror(through).join(through)

    def project(self, pt):
        """The orthogonal projection of a point onto the line.

        Parameters
        ----------
        pt : Point
            The point to project.

        Returns
        -------
        Point
            The projected point.

        """
        l = self.perpendicular(pt)
        return self.meet(l)

    @property
    def base_point(self):
        """Point: A base point for the line, arbitrarily chosen."""
        if self.dim > 2:
            return Point(self.basis_matrix[0, :])

        if np.isclose(self.array[2], 0):
            return Point(0, 0)

        if not np.isclose(self.array[1], 0):
            return Point([0, -self.array[2], self.array[1]])

        return Point([self.array[2], 0, -self.array[0]])

    @property
    def direction(self):
        """Point: The direction of the line (not normalized)."""
        if self.dim > 2:
            base = self.basis_matrix
            return Point(base[0, :]) - Point(base[1, :])
        if np.isclose(self.array[0], 0) and np.isclose(self.array[1], 0):
            return Point([0, 1, 0])
        return Point(self.array[1], -self.array[0])

    @property
    def basis_matrix(self):
        """numpy.ndarray: A matrix with orthonormal basis vectors as rows."""
        if self.dim == 2:
            a = self.base_point.array
            b = np.cross(self.array, a)
            result = np.array([a, b])
            return result / np.linalg.norm(result)
        return super(Line, self).basis_matrix

    def mirror(self, pt):
        """Construct the reflection of a point at this line.

        Parameters
        ----------
        pt : Point
            The point to reflect.

        Returns
        -------
        Point
            The mirror point.

        """
        l = self
        if self.dim >= 3:
            e = join(self, pt)
            m = e.basis_matrix
            m = m[np.argsort(np.abs(m.dot(pt.array)))]
            pt = Point(m.dot(pt.array))
            a, b = m.dot(self.basis_matrix.T).T
            l = Line(Point(a), Point(b))
        l1 = I.join(pt)
        l2 = J.join(pt)
        p1 = l.meet(l1)
        p2 = l.meet(l2)
        m1 = p1.join(J)
        m2 = p2.join(I)
        result = m1.meet(m2)
        if self.dim >= 3:
            return Point(m.T.dot(result.array))
        return result


infty = Line(0, 0, 1)


class Plane(Subspace):

    def __init__(self, *args):
        if all(isinstance(o, (Line, Point)) for o in args):
            super(Plane, self).__init__(join(*args))
        else:
            super(Plane, self).__init__(*args)

    @property
    def basis_matrix(self):
        """numpy.ndarray: A matrix with orthonormal basis vectors as rows."""
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
        """sympy.Poly: The polynomial defining this hyperplane."""
        return super(Plane, self).polynomials()[0]

    def __repr__(self):
        return "Plane({})".format(",".join(self.array.astype(str)))

    def mirror(self, pt):
        """Construct the reflection of a point at this plane.

        Currently only works in 3D.

        Parameters
        ----------
        pt : Point
            The point to reflect.

        Returns
        -------
        Point
            The mirror point.

        """
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
        """The orthogonal projection of a point onto the plane.

        Parameters
        ----------
        pt : Point
            The point to project.

        Returns
        -------
        Point
            The projected point.

        """
        l = self.perpendicular(pt)
        return self.meet(l)

    def perpendicular(self, through):
        """Construct the perpendicular line though a point.

        Parameters
        ----------
        through : Point
            The point through which the perpendicular is constructed.

        Returns
        -------
        Line
            The perpendicular line.

        """
        return self.mirror(through).join(through)


def infty_hyperplane(dimension):
    if dimension == 2:
        return infty
    return Plane([0] * dimension + [1])


infty_plane = infty_hyperplane(3)
