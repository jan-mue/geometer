import numpy as np

from .base import (EQ_TOL_ABS, LeviCivitaTensor, ProjectiveCollection, ProjectiveElement, Tensor, TensorCollection,
                   TensorDiagram)
from .exceptions import GeometryException, LinearDependenceError, NotCoplanar
from .utils import matmul, matvec, null_space


def _join_meet_duality(*args, intersect_lines=True, check_dependence=True, normalize_result=True):
    if len(args) < 2:
        raise ValueError("Expected at least 2 arguments, got %s." % len(args))

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
        if isinstance(a, (Line, LineCollection)) and isinstance(b, (Plane, PlaneCollection)) \
                or isinstance(b, (Line, LineCollection)) and isinstance(a, (Plane, PlaneCollection)):
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
            result = TensorDiagram(*[(e, a)] * a.tensor_shape[1], *[(e, b)] * (n - a.tensor_shape[1])).calculate()
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
                    max_ind = np.abs(array).reshape((np.prod(array.shape[: coplanar.ndim]), -1)).argmax(1)
                    i = np.unravel_index(max_ind, array.shape[coplanar.ndim:])
                    i = tuple(np.reshape(x, array.shape[: coplanar.ndim]) for x in i)
                    indices = tuple(np.indices(array.shape[: coplanar.ndim]))
                    if not intersect_lines:
                        result = array[indices + (i[0], Ellipsis)]
                        result_rank = result.ndim - coplanar.ndim
                        result = TensorCollection(result, covariant=False, tensor_rank=result_rank, copy=False)
                    else:
                        result = TensorCollection(array[indices + (slice(None),) + i[1:]], copy=False)

            elif intersect_lines or n == 4:
                # can't intersect lines that are not coplanar and can't join skew lines in 3D
                raise NotCoplanar("The given lines are not all coplanar.")
            elif np.any(coplanar) and (
                    isinstance(a, TensorCollection) or isinstance(b, TensorCollection)
            ):
                raise GeometryException(
                    "Can only join tensors that are either all coplanar or all not coplanar."
                )

        else:
            # TODO: intersect arbitrary subspaces (use GA)
            raise ValueError("Operation not supported.")

    else:
        raise ValueError("Wrong number of arguments.")

    if check_dependence:
        is_zero = result.is_zero()
        if len(result._collection_indices) == 0 and is_zero:
            raise LinearDependenceError("Arguments are not linearly independent.")
        elif np.any(is_zero):
            raise LinearDependenceError("Some arguments are not linearly independent.", is_zero)

    if isinstance(result, TensorCollection):

        if normalize_result:
            axes = tuple(result._covariant_indices) + tuple(result._contravariant_indices)
            max_abs = np.max(np.abs(result.array), axis=axes, keepdims=True)
            _, max_exponent = np.frexp(max_abs)
            result.array = _divide_by_power_of_two(result.array, max_exponent)

        if result.tensor_shape == (0, 1):
            return LineCollection(result, copy=False) if n == 3 else PlaneCollection(result, copy=False)
        if result.tensor_shape == (1, 0):
            return PointCollection(result, copy=False)
        if result.tensor_shape == (2, 0):
            return LineCollection(result, copy=False).contravariant_tensor
        if result.tensor_shape == (0, n - 2):
            return LineCollection(result, copy=False)

        return SubspaceCollection(result, copy=False)

    if normalize_result:
        # normalize result to avoid large values
        max_abs = np.max(np.abs(result.array))
        _, max_exponent = np.frexp(max_abs)
        if max_exponent != 0:
            result.array = _divide_by_power_of_two(result.array, max_exponent)

    if result.tensor_shape == (0, 1):
        return Line(result, copy=False) if n == 3 else Plane(result, copy=False)
    if result.tensor_shape == (1, 0):
        return Point(result, copy=False)
    if result.tensor_shape == (2, 0):
        return Line(result, copy=False).contravariant_tensor
    if result.tensor_shape == (0, n - 2):
        return Line(result, copy=False)

    return Subspace(result, copy=False)


def _divide_by_power_of_two(array, power):
    if array.dtype.kind == "c":
        rm, re = np.frexp(array.real)
        im, ie = np.frexp(array.imag)
        re -= power
        ie -= power
        out = np.empty_like(array)
        np.ldexp(rm, re, out=out.real)
        np.ldexp(im, ie, out=out.imag)
        return out

    mantissa, exponent = np.frexp(array)
    exponent -= power
    return np.ldexp(mantissa, exponent)


def join(*args, _check_dependence=True, _normalize_result=True):
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
    return _join_meet_duality(
        *args,
        intersect_lines=False,
        check_dependence=_check_dependence,
        normalize_result=_normalize_result
    )


def meet(*args, _check_dependence=True, _normalize_result=True):
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
    return _join_meet_duality(
        *args,
        intersect_lines=True,
        check_dependence=_check_dependence,
        normalize_result=_normalize_result
    )


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
    **kwargs
        Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    """

    def __init__(self, *args, **kwargs):
        if np.isscalar(args[0]):
            super(Point, self).__init__(*args, 1, **kwargs)
        else:
            super(Point, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        result = super(Point, self).__getitem__(index)

        if not isinstance(result, Tensor) or result.tensor_shape != (1, 0):
            return result

        return Point(result, copy=False)

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
        return "Point({})".format(", ".join(self.normalized_array[:-1].astype(str))) + (
            " at Infinity" if self.isinf else "")

    def _matrix_transform(self, m):
        return Point(np.dot(m, self.array), copy=False)

    @property
    def normalized_array(self):
        """numpy.ndarray: The coordinate array of the point with the last coordinate normalized to one."""
        if self.isinf or self.array[-1] == 1:
            return np.real_if_close(self.array)
        return np.real_if_close(self.array / self.array[-1])

    @property
    def lie_coordinates(self):
        """Point: The Lie coordinates of a point in 2D."""
        x = self.normalized_array[:-1]
        return Point([(1 + x.dot(x)) / 2, (1 - x.dot(x)) / 2, x[0], x[1], 0])

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
        """bool: true if the point lies at infinity i.e. the last coordinate is zero."""
        return np.isclose(self.array[-1], 0, atol=EQ_TOL_ABS)

    @property
    def isreal(self):
        """bool: true if the point is real, i.e. the imaginary part is zero for all components."""
        return np.all(np.isreal(self.normalized_array))


I = Point([-1j, 1, 0])
J = Point([1j, 1, 0])


class Subspace(ProjectiveElement):
    """Represents a general subspace of a projective space. Line and Plane are subclasses.

    Parameters
    ----------
    *args
        The coordinates of the subspace. Instead of separate coordinates, a single iterable can be supplied.
    **kwargs
        Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("covariant", False)
        super(Subspace, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        result = super(Subspace, self).__getitem__(index)

        if not isinstance(result, Tensor) or result.tensor_shape != self.tensor_shape:
            return result

        return type(self)(result, copy=False)

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
        return null_space(x, self.shape[-1] - self.rank).T

    @property
    def general_point(self):
        """Point: A point in general position i.e. not in the subspace, to be used in geometric constructions."""
        n = self.dim + 1
        p = Point(np.zeros(n, dtype=int), copy=False)
        for i in range(n):
            p[-i - 1] = 1
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
            raise TypeError("argument of type %s not supported" % type(other))

        axes = tuple(result._covariant_indices) + tuple(result._contravariant_indices)
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
    **kwargs
        Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

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
        if self.dim != 3:
            raise NotImplementedError("Expected dimension 3 but found dimension %s." % str(self.dim))
        if self.tensor_shape[0] > 0:
            return self
        e = LeviCivitaTensor(4)
        diagram = TensorDiagram((e, self), (e, self))
        return Line(diagram.calculate(), copy=False)

    @property
    def contravariant_tensor(self):
        """Line: The contravariant version of a line in 3D."""
        if self.dim != 3:
            raise NotImplementedError("Expected dimension 3 but found dimension %s." % str(self.dim))
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
        d = TensorDiagram(*[(e, self)] * (self.dim - 1), *[(e, other)] * (self.dim - 1))
        return d.calculate() == 0

    def perpendicular(self, through, plane=None):
        """Construct the perpendicular line though a point.

        Parameters
        ----------
        through : Point, PointCollection
            The point through which the perpendicular is constructed.
        plane : Plane, optional
            In three or higher dimensional spaces, the plane in which
            the perpendicular line is supposed to lie can be specified.

        Returns
        -------
        Line
            The perpendicular line.

        """
        result = LineCollection(np.expand_dims(self.array, 0), copy=False).perpendicular(through, plane=plane)
        return result if isinstance(through, PointCollection) else result[0]

    def project(self, pt):
        """The orthogonal projection of a point onto the line.

        Parameters
        ----------
        pt : Point, PointCollection
            The point(s) to project.

        Returns
        -------
        Point or PointCollection
            The projected point(s).

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

    def _matrix_transform(self, m):
        transformed_basis = np.dot(m, self.basis_matrix.T)
        transformed_basis = [Point(p, copy=False) for p in transformed_basis.T]
        return join(*transformed_basis)

    @property
    def lie_coordinates(self):
        """Point: The Lie coordinates of a line in 2D."""
        g = self.array
        return Point([-g[2], g[2], g[0], g[1], np.sqrt(g[:2].dot(g[:2]))])

    def mirror(self, pt):
        """Construct the reflection of a point at this line.

        Parameters
        ----------
        pt : Point, PointCollection
            The point(s) to reflect.

        Returns
        -------
        Point or PointCollection
            The mirror point(s).

        References
        ----------
        .. [1] J. Richter-Gebert: Perspectives on Projective Geometry, Section 19.1

        """
        result = LineCollection(np.expand_dims(self.array, 0), copy=False).mirror(pt)
        return result if isinstance(pt, PointCollection) else result[0]


infty = Line(0, 0, 1)


class Plane(Subspace):
    """Represents a hyperplane in a projective space of arbitrary dimension.

    Parameters
    ----------
    *args
        The points/lines spanning the plane or the coordinates of the hyperplane. Instead of separate coordinates, a
        single iterable can be supplied.
    **kwargs
        Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    """

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
        pt : Point, PointCollection
            The point(s) to reflect.

        Returns
        -------
        Point or PointCollection
            The mirror point(s).

        """
        result = PlaneCollection(np.expand_dims(self.array, 0), copy=False).mirror(pt)
        return result if isinstance(pt, PointCollection) else result[0]

    def project(self, pt):
        """The orthogonal projection of a point onto the plane.

        Only works in 3D.

        Parameters
        ----------
        pt : Point, PointCollection
            The point(s) to project.

        Returns
        -------
        Point or PointCollection
            The projected point(s).

        """
        l = self.perpendicular(pt)
        return self.meet(l)

    def perpendicular(self, through):
        """Construct the perpendicular line though a point or the perpendicular plane through a line.

        Only works in 3D.

        Parameters
        ----------
        through : Point, PointCollection, Line, LineCollection
            The point(s) or line(s) through which the perpendicular is constructed.

        Returns
        -------
        Line, LineCollection, Plane or PlaneCollection
            The perpendicular line(s) or plane(s).

        """
        result = PlaneCollection(
            np.expand_dims(self.array, 0), copy=False
        ).perpendicular(through)
        return result if isinstance(through, PointCollection) else result[0]


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

    def __repr__(self):
        return "{}({})".format(
            self.__class__.__name__, str(self.normalized_array.tolist())
        )

    @staticmethod
    def _normalize_array(array):
        isinf = np.isclose(array[..., -1], 0, atol=EQ_TOL_ABS)
        if np.all(isinf | (array[..., -1] == 1)):
            return np.real_if_close(array)
        result = array.astype(np.complex128)
        result[~isinf] /= array[~isinf, -1, None]
        return np.real_if_close(result)

    def _matrix_transform(self, m):
        return PointCollection(matvec(m, self.array), copy=False)

    @property
    def normalized_array(self):
        """numpy.ndarray: The coordinate array of the points with the last coordinates normalized to 1."""
        return self._normalize_array(self.array)

    @property
    def isinf(self):
        """array_like: Boolean array that indicates which points lie at infinity."""
        return np.isclose(self.array[..., -1], 0, atol=EQ_TOL_ABS)

    @property
    def isreal(self):
        """array_like: Boolean array that indicates which points are real."""
        return np.all(np.isreal(self.normalized_array), axis=-1)


class SubspaceCollection(ProjectiveCollection):
    """A collection of subspaces.

    Parameters
    ----------
    elements : array_like
        A sequence of Subspace objects, a numpy array, a Tensor or a (nested) sequence of numbers.
    tensor_rank : int, optional
        The rank of the tensors contained in the collection. Default is 1.
    **kwargs
        Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    """

    _element_class = Subspace

    def __init__(self, elements, *, tensor_rank=1, **kwargs):
        super(SubspaceCollection, self).__init__(
            elements, covariant=False, tensor_rank=tensor_rank, **kwargs
        )

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
        x = x.reshape(x.shape[: len(self._collection_indices)] + (-1, x.shape[-1]))
        return np.swapaxes(null_space(x, self.shape[-1] - self.rank + len(self._collection_indices)), -1, -2)

    def _matrix_transform(self, m):
        transformed_basis = matmul(self.basis_matrix, m, transpose_b=True)
        transformed_basis = [PointCollection(p, copy=False) for p in np.swapaxes(transformed_basis, 0, -2)]
        return join(*transformed_basis)

    @property
    def general_point(self):
        """PointCollection: Points in general position i.e. not in the subspaces."""
        n = self.dim + 1
        s = [self.shape[i] for i in self._collection_indices]
        p = PointCollection(np.zeros(s + [n], dtype=int), copy=False)
        ind = np.ones(s, dtype=bool)
        for i in range(n):
            p[ind, -i - 1] = 1
            ind = self.contains(p)
            if not np.any(ind):
                break
        return p

    def parallel(self, through):
        """Returns the subspaces through given points that are parallel to this collection of subspaces.

        Parameters
        ----------
        through : Point or PointCollection
            The point through which the parallel subspaces are to be constructed.

        Returns
        -------
        SubspaceCollection
            The parallel subspaces.

        """
        x = self.meet(infty_hyperplane(self.dim))
        return join(x, through)

    def contains(self, other, tol=EQ_TOL_ABS):
        """Tests whether given points or lines lie in the subspaces.

        Parameters
        ----------
        other : Point, PointCollection, Line or LineCollection
            The object(s) to test.
        tol : float, optional
            The accepted tolerance.

        Returns
        -------
        array_like
            Boolean array that indicates which of given points/lines lies in the subspaces.

        """
        if isinstance(other, (Point, PointCollection)):
            result = self * other

        elif isinstance(other, (Line, LineCollection)):
            result = self * other.covariant_tensor

        else:
            # TODO: test subspace
            raise ValueError("argument of type %s not supported" % type(other))

        axes = tuple(result._covariant_indices) + tuple(result._contravariant_indices)
        return np.all(np.isclose(result.array, 0, atol=tol), axis=axes)


class LineCollection(SubspaceCollection):
    """A collection of lines.

    Parameters
    ----------
    *args
        Two collections of points or a (nested) sequence of line coordinates.
    **kwargs
        Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

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

        if not isinstance(result, TensorCollection) or result.tensor_shape != (0, self.dim - 1):
            return result

        return LineCollection(result, copy=False)

    @property
    def base_point(self):
        """PointCollection: Base points for the lines, arbitrarily chosen."""
        base = self.basis_matrix
        p, q = base[..., 0, :], base[..., 1, :]
        isinf = np.isclose(p[..., -1, None], 0, atol=EQ_TOL_ABS)
        result = np.where(isinf, q, p)
        return PointCollection(result, copy=False)

    @property
    def direction(self):
        """PointCollection: The direction of the lines (not normalized)."""
        base = self.basis_matrix
        p, q = base[..., 0, :], base[..., 1, :]
        p_isinf = np.isclose(p[..., -1, None], 0, atol=EQ_TOL_ABS)
        q_isinf = np.isclose(q[..., -1, None], 0, atol=EQ_TOL_ABS)
        result = np.where(p_isinf, p, q)
        result = np.where(~(p_isinf | q_isinf), p / p[..., -1, None] - q / q[..., -1, None], result)
        return PointCollection(result, copy=False)

    @property
    def covariant_tensor(self):
        """LineCollection: The covariant tensors of lines in 3D."""
        if self.dim != 3:
            raise NotImplementedError("Expected dimension 3 but found dimension %s." % str(self.dim))
        if self.tensor_shape[0] > 0:
            return self
        e = LeviCivitaTensor(4)
        diagram = TensorDiagram((e, self), (e, self))
        return LineCollection(diagram.calculate(), copy=False)

    @property
    def contravariant_tensor(self):
        """LineCollection: The contravariant tensors of lines in 3D."""
        if self.dim != 3:
            raise NotImplementedError("Expected dimension 3 but found dimension %s." % str(self.dim))
        if self.tensor_shape[1] > 0:
            return self
        e = LeviCivitaTensor(4, False)
        diagram = TensorDiagram((self, e), (self, e))
        return LineCollection(diagram.calculate(), copy=False)

    def mirror(self, pt):
        """Construct the reflection of points at the lines.

        Parameters
        ----------
        pt : Point, PointCollection
            The point to reflect.

        Returns
        -------
        PointCollection
            The mirror points.

        References
        ----------
        .. [1] J. Richter-Gebert: Perspectives on Projective Geometry, Section 19.1

        """
        l = self
        if self.dim >= 3:
            # TODO: handle points on self
            e = join(self, pt)

            if self.dim == 3:
                f = e.perpendicular(self)
                return f.mirror(pt)

            m = e.basis_matrix
            arr = matvec(m, pt.array)
            arr_sort = np.argsort(np.abs(arr), axis=-1)
            arr_ind = tuple(np.indices(arr.shape)[:-1])
            m = m[arr_ind + (arr_sort, slice(None))]
            pt = PointCollection(arr[arr_ind + (arr_sort,)], copy=False)
            l = self._matrix_transform(m)
        l1 = join(I, pt, _normalize_result=False)
        l2 = join(J, pt, _normalize_result=False)
        p1 = l.meet(l1)
        p2 = l.meet(l2)
        m1 = join(p1, J, _normalize_result=False)
        m2 = join(p2, I, _normalize_result=False)
        result = m1.meet(m2)
        if self.dim >= 3:
            return result._matrix_transform(np.swapaxes(m, -1, -2))
        return result

    def perpendicular(self, through, plane=None):
        """Construct the perpendicular line though a point.

        Parameters
        ----------
        through : Point, PointCollection
            The point through which the perpendicular is constructed.
        plane : Plane, PlaneCollection, optional
            In three or higher dimensional spaces, the planes in which
            the perpendicular lines are supposed to lie can be specified.

        Returns
        -------
        LineCollection
            The perpendicular lines.

        """
        n = self.dim + 1
        contains = self.contains(through)
        result = LineCollection(np.empty(contains.shape + (n,) * (n - 2), np.complex128))
        if np.any(contains):
            l = self[contains] if len(self) > 1 else self

            if n > 3:

                if plane is None:
                    # additional point is required to determine the exact line
                    plane = join(l, l.general_point)
                elif isinstance(plane, PlaneCollection):
                    plane = plane[contains]

                basis = plane.basis_matrix
                line_pts = matmul(l.basis_matrix, basis, transpose_b=True)
                l = LineCollection(
                    np.cross(line_pts[..., 0, :], line_pts[..., 1, :]), copy=False
                )

            from .operators import harmonic_set

            p = l.meet(infty)
            q = harmonic_set(I, J, p)

            if n > 3:
                q = q._matrix_transform(np.swapaxes(basis, -1, -2))

            result[contains] = join(
                through if isinstance(through, Point) else through[contains], q
            )

        if np.any(~contains):
            if isinstance(through, PointCollection):
                through = through[~contains]
            if len(self) > 1:
                result[~contains] = self[~contains].mirror(through).join(through)
            else:
                result[~contains] = self.mirror(through).join(through)

        return LineCollection(np.real_if_close(result.array), copy=False)

    def project(self, pt):
        """The orthogonal projection of points onto the lines.

        Parameters
        ----------
        pt : Point, PointCollection
            The points to project.

        Returns
        -------
        PointCollection
            The projected points.

        """
        l = self.perpendicular(pt)
        return self.meet(l)


class PlaneCollection(SubspaceCollection):
    """A collection of planes.

    Parameters
    ----------
    *args
        The collections of points/lines spanning the planes or the coordinates of multiple hyperplanes.
    **kwargs
        Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

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

    @property
    def basis_matrix(self):
        # TODO: vectorize this
        return np.stack([e.basis_matrix for e in self], axis=0)

    def mirror(self, pt):
        """Construct the reflections of points at the planes.

        Only works in 3D.

        Parameters
        ----------
        pt : Point, PointCollection
            The points to reflect.

        Returns
        -------
        PointCollection
            The mirror points.

        """
        if self.dim != 3:
            raise NotImplementedError("Expected dimension 3 but found dimension %s." % str(self.dim))
        l = self.meet(infty_plane)
        basis = l.basis_matrix
        l = LineCollection(np.cross(basis[..., 0, :-1], basis[..., 1, :-1]), copy=False)
        p = l.base_point
        polars = LineCollection(p.array, copy=False)

        from .curve import absolute_conic

        p1, p2 = absolute_conic.intersect(polars)
        p1 = PointCollection(np.append(p1.array, np.zeros(p1.shape[:-1] + (1,)), axis=-1), copy=False)
        p2 = PointCollection(np.append(p2.array, np.zeros(p2.shape[:-1] + (1,)), axis=-1), copy=False)

        l1 = p1.join(pt)
        l2 = p2.join(pt)
        q1 = self.meet(l1)
        q2 = self.meet(l2)
        m1 = q1.join(p2)
        m2 = q2.join(p1)
        return m1.meet(m2)

    def project(self, pt):
        """The orthogonal projection of points onto the planes.

        Only works in 3D.

        Parameters
        ----------
        pt : Point, PointCollection
            The points to project.

        Returns
        -------
        PointCollection
            The projected points.

        """
        l = self.perpendicular(pt)
        return self.meet(l)

    def perpendicular(self, through):
        """Construct the perpendicular lines though the given points
        or the perpendicular planes through the given lines.

        Only works in 3D.

        Parameters
        ----------
        through : Point, PointCollection, Line, LineCollection
            The points or lines through which the perpendiculars are constructed.

        Returns
        -------
        LineCollection or PlaneCollection
            The perpendicular lines or planes.

        """
        if self.dim != 3:
            raise NotImplementedError("Expected dimension 3 but found dimension %s." % str(self.dim))
        contains = self.contains(through)
        result = LineCollection(np.empty(contains.shape + (4, 4), dtype=np.complex128), copy=False)
        if np.any(contains):
            from .curve import absolute_conic
            from .operators import harmonic_set

            if len(self) > 1:
                l = self[contains].meet(infty_plane)
            else:
                l = self.meet(infty_plane)
            basis = l.basis_matrix[..., :-1]
            l = LineCollection(np.cross(basis[..., 0, :], basis[..., 1, :]), copy=False)

            if isinstance(through, (Line, LineCollection)):
                if not np.all(contains):
                    raise ValueError("All lines must lie in the planes.")

                p = through.meet(infty_plane)
                if isinstance(through, Line):
                    polar = Line(p.array[:-1], copy=False)
                else:
                    polar = LineCollection(p.array[..., :-1], copy=False)
                tangent_points = absolute_conic.intersect(polar)
                q = harmonic_set(*tangent_points, l.meet(polar))
                q = PointCollection(np.append(q.array, np.zeros(q.shape[:-1] + (1,)), axis=-1), copy=False)
                return through.join(q)

            p1 = PointCollection(basis[..., 0, :], copy=False)
            p2 = PointCollection(basis[..., 1, :], copy=False)
            polar1 = LineCollection(p1.array, copy=False)
            polar2 = LineCollection(p2.array, copy=False)

            tangent_points1 = absolute_conic.intersect(polar1)
            tangent_points2 = absolute_conic.intersect(polar2)

            q1 = harmonic_set(*tangent_points1, l.meet(polar1))
            q2 = harmonic_set(*tangent_points2, l.meet(polar2))
            m1, m2 = p1.join(q1), p2.join(q2)

            p = m1.meet(m2)
            p = PointCollection(np.append(p.array, np.zeros(p.shape[:-1] + (1,)), axis=-1), copy=False)

            result[contains] = (through if isinstance(through, Point) else through[contains]).join(p)

        if np.any(~contains):
            if isinstance(through, PointCollection):
                through = through[~contains]
            if len(self) > 1:
                result[~contains] = self[~contains].mirror(through).join(through)
            else:
                result[~contains] = self.mirror(through).join(through)

        return LineCollection(np.real_if_close(result.array), copy=False)
