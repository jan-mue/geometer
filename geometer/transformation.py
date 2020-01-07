import numpy as np
from .base import ProjectiveElement, TensorDiagram, LeviCivitaTensor, Tensor
from .point import Point, Subspace
from .curve import Quadric, Conic
from .shapes import Polytope


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

    """
    if axis is None:
        return Transformation([[np.cos(angle), - np.sin(angle), 0],
                               [np.sin(angle), np.cos(angle), 0],
                               [0, 0, 1]])

    dimension = axis.dim
    e = LeviCivitaTensor(dimension, False)
    a = axis.normalized_array[:-1]
    a = a / np.linalg.norm(a)
    d = TensorDiagram(*[(Tensor(a), e) for _ in range(dimension - 2)])
    u = d.calculate().array
    v = np.outer(a, a)
    result = np.eye(dimension+1)
    result[:dimension, :dimension] = np.cos(angle)*np.eye(dimension) + np.sin(angle)*u + (1 - np.cos(angle))*v
    return Transformation(result)


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
    x = Point(*coordinates)
    m = np.eye(x.dim + 1)
    m[:-1, -1] = x.normalized_array[:-1]
    return Transformation(m)


def identity(dim):
    return Transformation(np.eye(dim+1))


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
        if isinstance(other, (Point, Transformation)):
            return type(other)(super(Transformation, self).__mul__(other))
        if isinstance(other, Subspace):
            return type(other)(other*self.inverse())
        if isinstance(other, Quadric):
            inv = self.inverse()
            cls = Conic if isinstance(other, Conic) else Quadric
            return cls(TensorDiagram((inv, other), (inv.copy(), other)).calculate())
        if isinstance(other, Polytope):
            return type(other)(other.array.dot(self.array.T))
        return NotImplemented

    def __mul__(self, other):
        return self.apply(other)

    def __pow__(self, power, modulo=None):
        if power == 0:
            return identity(self.dim)
        if power < 0:
            return self.inverse().__pow__(-power, modulo)
        return super(Transformation, self).__pow__(power, modulo)

    def inverse(self):
        """Calculates the inverse projective transformation.

        Returns
        -------
        Transformation
            The inverse transformation.

        """
        return Transformation(np.linalg.inv(self.array))
