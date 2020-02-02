import numpy as np
from .base import ProjectiveElement, TensorDiagram, LeviCivitaTensor, Tensor
from .point import Point, Subspace, infty_hyperplane


def identity(dim):
    """Returns the identity transformation.

    Parameters
    ----------
    dim : int
        The dimension of the projective space that the transformation acts on.

    Returns
    -------
    Transformation
        The identity transformation.

    """
    return Transformation(np.eye(dim+1))


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
    d = TensorDiagram(*[(Tensor(a), e) for _ in range(dimension - 2)])
    u = d.calculate().array
    v = np.outer(a, a)
    result = np.cos(angle)*np.eye(dimension) + np.sin(angle)*u + (1 - np.cos(angle))*v

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

    p = affine_transform(np.eye(axis.dim) - 2*np.outer(v, v.conj()))

    base = axis.basis_matrix
    ind = np.where(base[:, -1] != 0)[0][0]
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

    """

    def __init__(self, *args):
        super(Transformation, self).__init__(*args, covariant=[0])

    def __apply__(self, transformation):
        return Transformation(super(Transformation, transformation).__mul__(self))

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
        return cls(t2.dot(np.linalg.inv(t1)))

    def apply(self, other):
        """Apply the transformation to another object.

        Parameters
        ----------
        other : Point, Transformation, Subspace, Quadric or Polytope
            The object to apply the transformation to.

        Returns
        -------
        Point, Transformation, Subspace, Quadric or Polytope
            The result of applying this transformation to the supplied object.

        """
        if hasattr(other, '__apply__'):
            return other.__apply__(self)
        raise NotImplementedError("Object of type %s cannot be transformed." % str(type(other)))

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
        return Transformation(result)

    def inverse(self):
        """Calculates the inverse projective transformation.

        Returns
        -------
        Transformation
            The inverse transformation.

        """
        return Transformation(np.linalg.inv(self.array))
