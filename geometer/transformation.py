import numpy as np

from .base import LeviCivitaTensor, ProjectiveCollection, ProjectiveElement, Tensor, TensorDiagram
from .point import Point, infty_hyperplane
from .utils import inv


def identity(dim, collection_dims=None):
    """Returns the identity transformation.

    Parameters
    ----------
    dim : int
        The dimension of the projective space that the transformation acts on.
    collection_dims : tuple of int, optional
        Collection dimensions for a collection of identity transformations.
        By default only a single transformation is returned.

    Returns
    -------
    Transformation or TransformationCollection
        The identity transformation(s).

    """
    if collection_dims is not None:
        e = np.eye(dim + 1)
        e = e.reshape((1,) * len(collection_dims) + e.shape)
        e = np.tile(e, collection_dims + (1, 1))
        return TransformationCollection(e)
    return Transformation(np.eye(dim + 1))


def affine_transform(matrix=None, offset=0):
    """Returns a projective transformation for the given affine transformation.

    Parameters
    ----------
    matrix : array_like, optional
        The transformation matrix.
    offset : array_like or float, optional
        The translation.

    Returns
    -------
    Transformation
        The projective transformation that represents the affine transformation.

    """
    n = 2
    dtype = np.float32

    if not np.isscalar(offset):
        offset = np.array(offset)
        n = offset.shape[0] + 1
        dtype = offset.dtype

    if matrix is not None:
        matrix = np.array(matrix)
        n = matrix.shape[0] + 1
        dtype = np.find_common_type([dtype, matrix.dtype], [])

    result = np.eye(n, dtype=dtype)

    if matrix is not None:
        result[:-1, :-1] = matrix

    result[:-1, -1] = offset
    return Transformation(result)


def rotation(angle, axis=None):
    """Returns a projective transformation that represents a rotation by the specified angle (and axis).

    Parameters
    ----------
    angle : float
        The angle to rotate by.
    axis : Point, optional
        The axis to rotate around when rotating points in 3D.

    Returns
    -------
    Transformation
        The rotation.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle

    """
    if axis is None:
        return affine_transform([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]])

    dimension = axis.dim
    e = LeviCivitaTensor(dimension, False)
    a = axis.normalized_array[:-1]
    a = a / np.linalg.norm(a)
    d = TensorDiagram(*[(Tensor(a, copy=False), e) for _ in range(dimension - 2)])
    u = d.calculate().array
    v = np.outer(a, a)
    result = np.cos(angle) * np.eye(dimension) + np.sin(angle) * u + (1 - np.cos(angle)) * v

    return affine_transform(result)


def translation(*coordinates):
    """Returns a projective transformation that represents a translation by the given coordinates.

    Parameters
    ----------
    *coordinates
        The coordinates by which points are translated when applying the resulting transformation.

    Returns
    -------
    Transformation
        The translation.

    """
    offset = Point(*coordinates)
    return affine_transform(offset=offset.normalized_array[:-1])


def scaling(*factors):
    """Returns a projective transformation that represents general scaling by given factors in each dimension.

    Parameters
    ----------
    *factors
        The scaling factors by which each dimension is scaled.

    Returns
    -------
    Transformation
        The scaling transformation.

    """
    if len(factors) == 1:
        factors = factors[0]
    return affine_transform(np.diag(factors))


def reflection(axis):
    """Returns a projective transformation that represents a reflection at the given axis/hyperplane.

    Parameters
    ----------
    axis : Subspace
        The 2D-line or hyperplane to reflect points at.

    Returns
    -------
    Transformation
        The reflection.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Householder_transformation

    """
    if axis == infty_hyperplane(axis.dim):
        return identity(axis.dim)

    v = axis.array[:-1]
    v = v / np.linalg.norm(v)

    p = affine_transform(np.eye(axis.dim) - 2 * np.outer(v, v.conj()))

    base = axis.basis_matrix
    ind = base[:, -1].nonzero()[0][0]
    x = base[ind, :-1] / base[ind, -1]
    x = Point(*x)

    return translation(x) * p * translation(-x)


class Transformation(ProjectiveElement):
    """Represents a projective transformation in an arbitrary projective space.

    The underlying array is the matrix representation of the projective transformation. The matrix must be
    a nonsingular square matrix of size n+1 when n is the dimension of the projective space.
    The transformation can be applied to a point or another object by multiplication.

    Parameters
    ----------
    *args
        The array that defines the matrix representing the transformation.
    **kwargs
        Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("covariant", [0])
        super(Transformation, self).__init__(*args, **kwargs)

    def __apply__(self, transformation):
        return Transformation(transformation.array.dot(self.array), copy=False)

    @classmethod
    def from_points(cls, *args):
        """Constructs a projective transformation in n-dimensional projective space from the image of n + 2 points in
        general position.

        For two dimensional transformations, 4 pairs of points are required, of which no three points are collinear.
        For three dimensional transformations, 5 pairs of points are required, of which no four points are coplanar.

        Parameters
        ----------
        *args
            Pairs of points, where in each pair one point is mapped to the other.

        Returns
        -------
        Transformation
            The transformation mapping each of the given points to the specified points.

        References
        ----------
        .. [1] J. Richter-Gebert: Perspectives on Projective Geometry, Proof of Theorem 3.4

        """
        a = [x.array for x, y in args]
        b = [y.array for x, y in args]
        m1 = np.column_stack(a[:-1])
        m2 = np.column_stack(b[:-1])
        d1 = np.linalg.solve(m1, a[-1])
        d2 = np.linalg.solve(m2, b[-1])
        t1 = m1.dot(np.diag(d1))
        t2 = m2.dot(np.diag(d2))

        return Transformation(t2.dot(np.linalg.inv(t1)))

    @classmethod
    def from_points_and_conics(cls, points1, points2, conic1, conic2):
        """Constructs a projective transformation from two conics and the image of pairs of 3 points on the conics.

        Parameters
        ----------
        points1 : list of Point
            Source points on conic1.
        points2 : list of Point
            Target points on conic2.
        conic1 : Conic
            Source quadric.
        conic2 : Conic
            Target quadric.

        Returns
        -------
        Transformation
            The transformation that maps the points to each other and the conic to the other conic.

        References
        ----------
        .. [1] https://math.stackexchange.com/questions/654275/homography-between-ellipses

        """
        a1, b1, c1 = points1
        l1, l2 = conic1.tangent(a1), conic1.tangent(b1)
        m = l1.meet(l2).join(c1)
        p, q = conic1.intersect(m)
        d1 = p if q == c1 else q

        a2, b2, c2 = points2
        l1, l2 = conic2.tangent(a2), conic2.tangent(b2)
        m = l1.meet(l2).join(c2)
        p, q = conic2.intersect(m)
        d2 = p if q == c2 else q

        return Transformation.from_points((a1, a2), (b1, b2), (c1, c2), (d1, d2))

    def apply(self, other):
        """Apply the transformation to another object.

        Parameters
        ----------
        other : Tensor
            The object to apply the transformation to.

        Returns
        -------
        Tensor
            The result of applying this transformation to the supplied object.

        """
        if hasattr(other, "__apply__"):
            return other.__apply__(self)
        raise NotImplementedError("Object of type %s cannot be transformed." % type(other))

    def __mul__(self, other):
        try:
            return self.apply(other)
        except NotImplementedError:
            return super(Transformation, self).__mul__(other)

    def __pow__(self, power, modulo=None):
        if power == 0:
            return identity(self.dim)
        if power < 0:
            return self.inverse().__pow__(-power, modulo)

        result = super(Transformation, self).__pow__(power, modulo)
        return Transformation(result, copy=False)

    def inverse(self):
        """Calculates the inverse projective transformation.

        Returns
        -------
        Transformation
            The inverse transformation.

        """
        return Transformation(np.linalg.inv(self.array))


class TransformationCollection(ProjectiveCollection):
    """A Collection of transformations."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("covariant", [0])
        super(TransformationCollection, self).__init__(*args, tensor_rank=2, **kwargs)

    def apply(self, other):
        """Apply the transformations to another object.

        Parameters
        ----------
        other : Tensor
            The object to apply the transformations to.

        Returns
        -------
        TensorCollection
            The result of applying the transformations to the supplied object.

        """
        if hasattr(other, "__apply__"):
            return other.__apply__(self)
        raise NotImplementedError("Object of type %s cannot be transformed." % type(other))

    def __pow__(self, power, modulo=None):
        if power == 0:
            return identity(self.dim, self.shape[: len(self._collection_indices)])
        if power < 0:
            return self.inverse().__pow__(-power, modulo)

        result = super(TransformationCollection, self).__pow__(power, modulo)
        return TransformationCollection(result, copy=False)

    def inverse(self):
        """Calculates the inverse projective transformations.

        Returns
        -------
        TransformationCollection
            The inverse transformations.

        """
        return TransformationCollection(inv(self.array))
