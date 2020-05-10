import numpy as np

from .base import ProjectiveElement, ProjectiveCollection, TensorDiagram, LeviCivitaTensor, TensorCollection, Tensor, EQ_TOL_ABS
from .exceptions import LinearDependenceError, NotCoplanar, GeometryException
from .utils import null_space


def _join_meet_duality(*args, intersect_lines=True):
    if len(args) < 2:
        raise ValueError("Expected at least 2 arguments, got %s." % str(len(args)))

    n = args[0].dim + 1

    # all arguments are 1-tensors, i.e. points or hypersurfaces (=lines in 2D)
    if all(o.tensor_shape == args[0].tensor_shape for o in args[1:]) and sum(args[0].tensor_shape) == 1:
        covariant = args[0].tensor_shape[0] > 0
        e = LeviCivitaTensor(n, not covariant)

        # summing all arguments with e gives a (n-len(args))-tensor (e.g. a 1-tensor for two 2D-lines)
        result = TensorDiagram(*[(o, e) if covariant else (e, o) for o in args]).calculate()

    # two lines/planes
    elif len(args) == 2:
        a, b = args
        if isinstance(a, (Line, LineCollection)) and isinstance(b, (Plane, PlaneCollection)) or \
                isinstance(b, (Line, LineCollection)) and isinstance(a, (Plane, PlaneCollection)):
            e = LeviCivitaTensor(n)
            result = TensorDiagram(*[(e, a)] * a.tensor_shape[1], *[(e, b)] * b.tensor_shape[1]).calculate()
        elif isinstance(a, (Subspace, SubspaceCollection)) and isinstance(b, (Point, PointCollection)):
            result = a * b
        elif isinstance(a, (Point, PointCollection)) and isinstance(b, (Subspace, SubspaceCollection)):
            result = b * a
        elif isinstance(a, (Line, LineCollection)) and isinstance(b, (Line, LineCollection)):
            # can assume that n >= 4, because for n = 3 lines are 1-tensors
            e = LeviCivitaTensor(n)

            # if this is zero, the lines are coplanar
            result = TensorDiagram(*[(e, a)] * a.tensor_shape[1], *[(e, b)] * (n-a.tensor_shape[1])).calculate()
            coplanar = result.is_zero()

            if np.all(coplanar):
                # this part is inspired by Jim Blinn, Lines in Space: A Tale of Two Lines
                diagram = TensorDiagram(*[(e, a)] * a.tensor_shape[1], (e, b))
                array = diagram.calculate().array

                if np.isscalar(coplanar):
                    i = np.unravel_index(np.abs(array).argmax(), array.shape)
                    if not intersect_lines:
                        # extract the common subspace
                        result = Tensor(array[i[0], ...], covariant=False, copy=False)
                    else:
                        # extract the point of intersection
                        result = Tensor(array[(slice(None),) + i[1:]], copy=False)
                else:
                    max_ind = np.abs(array).reshape((np.prod(array.shape[:coplanar.ndim]), -1)).argmax(1)
                    i = np.unravel_index(max_ind, array.shape[coplanar.ndim:])
                    i = tuple(np.reshape(x, array.shape[:coplanar.ndim]) for x in i)
                    indices = tuple(np.indices(array.shape[:coplanar.ndim]))
                    if not intersect_lines:
                        result = array[indices + (i[0], Ellipsis)]
                        result = TensorCollection(result, covariant=False, tensor_rank=result.ndim-coplanar.ndim, copy=False)
                    else:
                        result = TensorCollection(array[indices + (slice(None),) + i[1:]], copy=False)

            elif intersect_lines or n == 4:
                # can't intersect lines that are not coplanar and can't join skew lines in 3D
                raise NotCoplanar("The given lines are not all coplanar.")
            elif np.any(coplanar) and (isinstance(a, TensorCollection) or isinstance(b, TensorCollection)):
                raise GeometryException("Can only join tensors that are either all coplanar or all not coplanar.")

        else:
            # TODO: intersect arbitrary subspaces (use GA)
            raise ValueError("Operation not supported.")

    else:
        raise ValueError("Wrong number of arguments.")

    if np.any(result.is_zero()):
        raise LinearDependenceError("Arguments are not linearly independent.")

    if isinstance(result, TensorCollection):

        axes = tuple(result._covariant_indices) + tuple(result._contravariant_indices)
        result.array = result.array / np.max(np.abs(result.array), axis=axes, keepdims=True)

        if result.tensor_shape == (0, 1):
            return LineCollection(result, copy=False) if n == 3 else PlaneCollection(result, copy=False)
        if result.tensor_shape == (1, 0):
            return PointCollection(result, copy=False)
        if result.tensor_shape == (2, 0):
            return LineCollection(result, copy=False).contravariant_tensor
        if result.tensor_shape == (0, n - 2):
            return LineCollection(result, copy=False)

        return SubspaceCollection(result, copy=False)

    # normalize result to avoid large values
    result.array = result.array / np.max(np.abs(result.array))

    if result.tensor_shape == (0, 1):
        return Line(result, copy=False) if n == 3 else Plane(result, copy=False)
    if result.tensor_shape == (1, 0):
        return Point(result, copy=False)
    if result.tensor_shape == (2, 0):
        return Line(result, copy=False).contravariant_tensor
    if result.tensor_shape == (0, n-2):
        return Line(result, copy=False)

    return Subspace(result, copy=False)


def join(*args):
    """Joins a number of objects to form a line, plane or subspace.

    Parameters
    ----------
    *args
        Objects to join, e.g. 2 points, lines, a point and a line or 3 points.

    Returns
    -------
    Subspace or SubspaceCollection
        The resulting line, plane or subspace.

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
    Point, PointCollection, Subspace or SubspaceCollection
        The resulting point, line or subspace.

    """
    return _join_meet_duality(*args, intersect_lines=True)


class Point(ProjectiveElement):
    """Represents points in a projective space of arbitrary dimension.

    The number of supplied coordinates determines the dimension of the space that the point lives in.
    If the coordinates are given as arguments (not in a single iterable), the coordinates will automatically be
    transformed into homogeneous coordinates, i.e. a one added as an additional coordinate.

    Addition and subtraction of finite and infinite points will always give a finite result if one of the points
    was finite beforehand.

    Parameters
    ----------
    *args
        A single iterable object or tensor or multiple (affine) coordinates.

    """

    def __init__(self, *args, **kwargs):
        if np.isscalar(args[0]):
            super(Point, self).__init__(*args, 1, **kwargs)
        else:
            super(Point, self).__init__(*args, **kwargs)

    def __add__(self, other):
        if isinstance(other, PointCollection):
            return NotImplemented
        if not isinstance(other, Point):
            return super(Point, self).__add__(other)
        a, b = self.normalized_array, other.normalized_array
        result = a[:-1] + b[:-1]
        result = np.append(result, max(a[-1], b[-1]))
        return Point(result, copy=False)

    def __sub__(self, other):
        if isinstance(other, PointCollection):
            return NotImplemented
        if not isinstance(other, Point):
            return super(Point, self).__sub__(other)
        a, b = self.normalized_array, other.normalized_array
        result = a[:-1] - b[:-1]
        result = np.append(result, max(a[-1], b[-1]))
        return Point(result, copy=False)

    def __mul__(self, other):
        if not np.isscalar(other):
            return super(Point, self).__mul__(other)
        result = self.normalized_array[:-1] * other
        result = np.append(result, self.array[-1] and 1)
        return Point(result, copy=False)

    def __truediv__(self, other):
        if not np.isscalar(other):
            return super(Point, self).__truediv__(other)
        result = self.normalized_array[:-1] / other
        result = np.append(result, self.array[-1] and 1)
        return Point(result, copy=False)

    def __repr__(self):
        return "Point({})".format(", ".join(self.normalized_array[:-1].astype(str))) + (" at Infinity" if self.isinf else "")

    @property
    def normalized_array(self):
        """numpy.ndarray: The normalized coordinates as array."""
        if self.isinf:
            return np.real_if_close(self.array)
        return np.real_if_close(self.array / self.array[-1])

    @property
    def lie_coordinates(self):
        """Point: The Lie coordinates of a point in 2D."""
        x = self.normalized_array[:-1]
        return Point([(1+x.dot(x))/2, (1-x.dot(x))/2, x[0], x[1], 0])

    def join(self, *others):
        """Execute the join of this point with other objects.

        Parameters
        ----------
        *others
            The objects to join the point with.

        Returns
        -------
        Subspace
            The result of the join operation.

        See Also
        --------
        join

        """
        return join(self, *others)

    @property
    def isinf(self):
        return np.isclose(self.array[-1], 0, atol=EQ_TOL_ABS)


I = Point([-1j, 1, 0])
J = Point([1j, 1, 0])


class Subspace(ProjectiveElement):
    """Represents a general subspace of a projective space. Line and Plane are subclasses.

    Parameters
    ----------
    *args
        The coordinates of the subspace. Instead of all coordinates separately, a single iterable can also be supplied.

    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("covariant", False)
        super(Subspace, self).__init__(*args, **kwargs)

    def __add__(self, other):
        if not isinstance(other, Point):
            return super(Subspace, self).__add__(other)

        from .transformation import translation
        return translation(other).apply(self)

    def __sub__(self, other):
        return self + (-other)

    @property
    def basis_matrix(self):
        """numpy.ndarray: A matrix with orthonormal basis vectors as rows."""
        x = self.array
        if x.ndim > 2:
            x = self.array.reshape(-1, x.shape[-1])
        return null_space(x).T

    @property
    def general_point(self):
        """Point: A point in general position i.e. not in the subspace, to be used in geometric constructions."""
        n = self.dim + 1
        arr = np.zeros(n, dtype=int)
        for i in range(n):
            arr[-i - 1] = 1
            p = Point(arr, copy=False)
            if not self.contains(p):
                return p

    def contains(self, other, tol=EQ_TOL_ABS):
        """Tests whether a given point or line lies in the subspace.

        Parameters
        ----------
        other : Point or Line
            The object to test.
        tol : float, optional
            The accepted tolerance.

        Returns
        -------
        bool
            True, if the given point/line lies in the subspace.

        """
        if isinstance(other, (Point, PointCollection)):
            result = self * other

        elif isinstance(other, (Line, LineCollection)):
            result = self * other.covariant_tensor

        else:
            # TODO: test subspace
            raise ValueError("argument of type %s not supported" % str(type(other)))

        axes = tuple(set(range(result.rank)) - other._collection_indices)
        return np.all(np.isclose(result.array, 0, atol=tol), axis=axes)

    def meet(self, *others):
        """Intersect the subspace with other objects.

        Parameters
        ----------
        *others
            The objects to intersect the subspace with.

        Returns
        -------
        Point or Subspace
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
        Subspace
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

    def __init__(self, *args, **kwargs):
        if len(args) == 2:
            kwargs["copy"] = False
            super(Line, self).__init__(join(*args), **kwargs)
        else:
            super(Line, self).__init__(*args, **kwargs)

    @property
    def covariant_tensor(self):
        """Line: The covariant version of a line in 3D."""
        if self.tensor_shape[0] > 0:
            return self
        e = LeviCivitaTensor(4)
        diagram = TensorDiagram((e, self), (e, self))
        return Line(diagram.calculate(), copy=False)

    @property
    def contravariant_tensor(self):
        """Line: The contravariant version of a line in 3D."""
        if self.tensor_shape[1] > 0:
            return self
        e = LeviCivitaTensor(4, False)
        diagram = TensorDiagram((self, e), (self, e))
        return Line(diagram.calculate(), copy=False)

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

        References
        ----------
        .. [1] Jim Blinn, Lines in Space: Back to the Diagrams, Line Intersections

        """
        if self.dim == 2:
            return True

        e = LeviCivitaTensor(self.dim + 1)
        d = TensorDiagram(*[(e, self)]*(self.dim - 1), *[(e, other)]*(self.dim - 1))
        return d.calculate() == 0

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
                e = join(self, self.general_point)

                basis = e.basis_matrix
                line_pts = basis.dot(self.basis_matrix.T)
                l = Line(np.cross(*line_pts.T), copy=False)

            from .operators import harmonic_set
            p = l.meet(infty)
            q = harmonic_set(I, J, p)

            if n > 3:
                q = Point(basis.T.dot(q.array), copy=False)

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
            base = self.basis_matrix
            p, q = Point(base[0, :], copy=False), Point(base[1, :], copy=False)
            if p.isinf:
                return q
            return p

        if np.isclose(self.array[2], 0, atol=EQ_TOL_ABS):
            return Point(0, 0)

        if not np.isclose(self.array[1], 0, atol=EQ_TOL_ABS):
            return Point([0, -self.array[2], self.array[1]])

        return Point([self.array[2], 0, -self.array[0]])

    @property
    def direction(self):
        """Point: The direction of the line (not normalized)."""
        if self.dim > 2:
            base = self.basis_matrix
            p, q = Point(base[0, :], copy=False), Point(base[1, :], copy=False)
            if p.isinf:
                return p
            if q.isinf:
                return q
            return Point(p.normalized_array - q.normalized_array, copy=False)

        if np.isclose(self.array[0], 0, atol=EQ_TOL_ABS) and np.isclose(self.array[1], 0, atol=EQ_TOL_ABS):
            return Point([0, 1, 0])

        return Point([self.array[1], -self.array[0], 0])

    @property
    def basis_matrix(self):
        """numpy.ndarray: A matrix with orthonormal basis vectors as rows."""
        if self.dim == 2:
            a = self.base_point.array
            b = np.cross(self.array, a)
            return np.array([a / np.linalg.norm(a), b / np.linalg.norm(b)])
        return super(Line, self).basis_matrix

    @property
    def lie_coordinates(self):
        """Point: The Lie coordinates of a line in 2D."""
        g = self.array
        return Point([-g[2], g[2], g[0], g[1], np.sqrt(g[:2].dot(g[:2]))])

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

        References
        ----------
        .. [1] J. Richter-Gebert: Perspectives on Projective Geometry, Section 19.1

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

    def __init__(self, *args, **kwargs):
        if all(isinstance(o, (Line, Point)) for o in args):
            kwargs["copy"] = False
            super(Plane, self).__init__(join(*args), **kwargs)
        else:
            super(Plane, self).__init__(*args, **kwargs)

    @property
    def basis_matrix(self):
        """numpy.ndarray: A matrix with orthonormal basis vectors as rows."""
        n = self.dim + 1
        i = self.array.nonzero()[0][0]
        result = np.zeros((n, n - 1), dtype=self.dtype)
        a = [j for j in range(n) if j != i]
        result[i, :] = self.array[a]
        result[a, range(n - 1)] = -self.array[i]
        q, r = np.linalg.qr(result)
        return q.T

    def __repr__(self):
        return "Plane({})".format(",".join(self.array.astype(str)))

    def mirror(self, pt):
        """Construct the reflection of a point at this plane.

        Only works in 3D.

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
        l = Line(np.cross(*l.basis_matrix[:, :-1]), copy=False)
        p = l.base_point
        polar = Line(p.array, copy=False)

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

        Only works in 3D.

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

        Only works in 3D.

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
            l = self.meet(infty_plane)
            l = Line(np.cross(*l.basis_matrix[:, :-1]), copy=False)
            p1, p2 = [Point(a) for a in l.basis_matrix]
            polar1 = Line(p1.array, copy=False)
            polar2 = Line(p2.array, copy=False)

            from .curve import absolute_conic
            tangent_points1 = absolute_conic.intersect(polar1)
            tangent_points2 = absolute_conic.intersect(polar2)

            from .operators import harmonic_set
            q1, q2 = harmonic_set(*tangent_points1, l.meet(polar1)), harmonic_set(*tangent_points2, l.meet(polar2))
            m1, m2 = p1.join(q1), p2.join(q2)

            p = m1.meet(m2)
            p = Point(np.append(p.array, 0))

            return through.join(p)

        return self.mirror(through).join(through)


def infty_hyperplane(dimension):
    if dimension == 2:
        return infty
    return Plane([0] * dimension + [1])


infty_plane = infty_hyperplane(3)


class PointCollection(ProjectiveCollection):
    """A collection of points.

    Parameters
    ----------
    elements : array_like
        A (nested) sequence of points or a numpy array that contains the coordinates of multiple points.
    homogenize : bool, optional
        If True, all points in the array will be converted to homogeneous coordinates, i.e. 1 will be added to
        the coordinates of each point in elements. By default homogenize is False.

    """
    _element_class = Point

    def __init__(self, elements, *, homogenize=False, **kwargs):
        if not isinstance(elements, (np.ndarray, Tensor)):
            elements = np.asarray(elements)
        super(PointCollection, self).__init__(elements, **kwargs)
        if homogenize is True:
            self.array = np.append(self.array, np.ones(self.shape[:-1] + (1,), self.dtype), axis=-1)

    def __add__(self, other):
        if not isinstance(other, (Point, PointCollection)):
            return super(PointCollection, self).__add__(other)
        a, b = self.normalized_array, other.normalized_array
        result = a[..., :-1] + b[..., :-1]
        result = np.append(result, np.maximum(a[..., -1:], b[..., -1:]), axis=-1)
        return PointCollection(result, copy=False)

    def __sub__(self, other):
        if not isinstance(other, (Point, PointCollection)):
            return super(PointCollection, self).__add__(other)
        a, b = self.normalized_array, other.normalized_array
        result = a[..., :-1] - b[..., :-1]
        result = np.append(result, np.maximum(a[..., -1:], b[..., -1:]), axis=-1)
        return PointCollection(result, copy=False)

    def __mul__(self, other):
        if not np.isscalar(other):
            return super(PointCollection, self).__mul__(other)
        result = self.normalized_array[..., :-1] * other
        result = np.append(result, self.array[..., -1:] != 0, axis=-1)
        return PointCollection(result, copy=False)

    def __truediv__(self, other):
        if not np.isscalar(other):
            return super(PointCollection, self).__truediv__(other)
        result = self.normalized_array[..., :-1] / other
        result = np.append(result, self.array[..., -1:] != 0, axis=-1)
        return PointCollection(result, copy=False)

    def join(self, *others):
        return join(self, *others)

    def __getitem__(self, index):
        result = super(PointCollection, self).__getitem__(index)

        if not isinstance(result, TensorCollection) or result.tensor_shape != (1, 0):
            return result

        return PointCollection(result, copy=False)

    @staticmethod
    def _normalize_array(array):
        isinf = np.isclose(array[..., -1], 0, atol=EQ_TOL_ABS)
        result = array.astype(np.complex128)
        result[~isinf] /= array[~isinf, -1, None]
        return np.real_if_close(result)

    @property
    def normalized_array(self):
        return self._normalize_array(self.array)


class SubspaceCollection(ProjectiveCollection):
    """A collection of subspaces.

    """
    _element_class = Subspace

    def __init__(self, *args, **kwargs):
        super(SubspaceCollection, self).__init__(*args, covariant=False, **kwargs)

    def meet(self, other):
        return meet(self, other)

    def join(self, *others):
        return join(self, *others)

    def __getitem__(self, index):
        result = super(SubspaceCollection, self).__getitem__(index)

        if not isinstance(result, TensorCollection) or result.tensor_shape == (0, 0):
            return result

        return SubspaceCollection(result, copy=False)

    @property
    def basis_matrix(self):
        x = self.array
        x = x.reshape(x.shape[:len(self._collection_indices)] + (-1, x.shape[-1]))
        return np.swapaxes(null_space(x), -1, -2)

    def contains(self, other, tol=EQ_TOL_ABS):
        if isinstance(other, (Point, PointCollection)):
            result = self * other

        elif isinstance(other, (Line, LineCollection)):
            result = self * other.covariant_tensor

        else:
            # TODO: test subspace
            raise ValueError("argument of type %s not supported" % str(type(other)))

        axes = tuple(set(range(result.rank)) - self._collection_indices)
        return np.all(np.isclose(result.array, 0, atol=tol), axis=axes)


class LineCollection(SubspaceCollection):
    """A collection of lines.

    """
    _element_class = Line

    def __init__(self, *args, **kwargs):
        if len(args) == 2:
            kwargs["copy"] = False
            super(LineCollection, self).__init__(join(*args), tensor_rank=-2, **kwargs)
        else:
            super(LineCollection, self).__init__(*args, tensor_rank=-2, **kwargs)

    def __getitem__(self, index):
        result = super(LineCollection, self).__getitem__(index)

        if not isinstance(result, TensorCollection) or result.tensor_shape != (0, self.dim-1):
            return result

        return LineCollection(result, copy=False)

    @property
    def covariant_tensor(self):
        if self.tensor_shape[0] > 0:
            return self
        e = LeviCivitaTensor(4)
        diagram = TensorDiagram((e, self), (e, self))
        return LineCollection(diagram.calculate(), copy=False)

    @property
    def contravariant_tensor(self):
        if self.tensor_shape[1] > 0:
            return self
        e = LeviCivitaTensor(4, False)
        diagram = TensorDiagram((self, e), (self, e))
        return LineCollection(diagram.calculate(), copy=False)


class PlaneCollection(SubspaceCollection):
    """A collection of planes.

    """
    _element_class = Plane

    def __init__(self, *args, **kwargs):
        if len(args) > 1:
            kwargs["copy"] = False
            super(PlaneCollection, self).__init__(join(*args), **kwargs)
        else:
            super(PlaneCollection, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        result = super(PlaneCollection, self).__getitem__(index)

        if not isinstance(result, TensorCollection) or result.tensor_shape != (0, 1):
            return result

        return PlaneCollection(result, copy=False)
