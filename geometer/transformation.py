import numpy as np
from .base import ProjectiveElement, TensorDiagram, LeviCivitaTensor, Tensor
from .point import Point, Line, Plane
from .curve import Quadric


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
    m = np.eye(len(coordinates) + 1)
    m[:-1, -1] = coordinates
    return Transformation(m)


class Transformation(ProjectiveElement):
    """Represents a projective transformation in an arbitrary projective space.

    The underlying array of the transformation is the matrix representation of the projective transformation.
    The transformation can be applied to points by simple multiplication.

    Parameters
    ----------
    *args
        The array that defines the matrix representing the transformation.

    """
    
    def __init__(self, *args):
        super(Transformation, self).__init__(*args, covariant=[0])

    @classmethod
    def from_points(cls, *args):
        """Constructs a projective transformation in n-dimensional projective space from the image of n + 2 points.

        For two dimensional transformations, 4 pairs of points are required.
        For three dimensional transformations, 5 pairs of points are required.

        Parameters
        ----------
        *args
            Pairs of points, where in each pair one point is mapped to the other.

        Returns
        -------
        Transformation
            The transformation mapping each of the given points to the specified points.

        """
        a = [x.array for x, y in args]
        b = [y.array for x, y in args]
        m1 = np.array(b[:-1]).T.dot(np.diag(b[-1]))
        m2 = np.array(a[:-1]).T.dot(np.diag(a[-1]))
        return Transformation(m1.dot(np.linalg.inv(m2)))

    def __mul__(self, other):
        if isinstance(other, (Point, Transformation)):
            return type(other)(super(Transformation, self).__mul__(other))
        if isinstance(other, (Line, Plane)):
            return type(other)(other*self.inverse())
        if isinstance(other, Quadric):
            inv = self.inverse()
            return type(other)(TensorDiagram((inv, other), (other, inv)).calculate())
        raise NotImplemented

    def __pow__(self, power, modulo=None):
        return Transformation(pow(self.array, power, modulo))

    def inverse(self):
        """Calculates the inverse projective transformation.

        Returns
        -------
        Transformation
            The inverse transformation.

        """
        return Transformation(np.linalg.inv(self.array))

    def __rmul__(self, other):
        return self*other
