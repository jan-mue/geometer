import numpy as np
import numpy.typing as npt


class GeometryException(Exception):
    """A general geometric error occurred."""


class TensorComputationError(GeometryException):
    """An error during a tensor computation occurred."""


class NotCollinear(GeometryException, ValueError):
    """The given values are not collinear."""


class NotCoplanar(GeometryException, ValueError):
    """The given values are not coplanar."""


class NotConcurrent(GeometryException, ValueError):
    """The given values are not concurrent."""


class NoIncidence(GeometryException, ValueError):
    """The given point is not incident with the subspace."""


class IncidenceError(GeometryException, ValueError):
    """The given point is incident with the subspace."""


class LinearDependenceError(GeometryException, ValueError):
    """The given values were linearly dependent, making the computation impossible.

    Attributes:
        dependent_values (numpy.ndarray): The indices of the sets of linearly dependent vectors.

    """

    def __init__(self, message: str, dependent_values: npt.ArrayLike = True) -> None:
        super().__init__(message)
        self.dependent_values = np.asarray(dependent_values)


class NotReducible(GeometryException, ValueError):
    """The given geometric object is not reducible."""


class IncompatibleShapeError(ValueError):
    """The given tensor has a shape that is not compatible."""
