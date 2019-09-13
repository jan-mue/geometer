

class GeometryException(Exception):
    """A general geometric error occurred."""


class TensorComputationError(GeometryException):
    """An error during a tensor computation occurred."""


class NotCollinear(GeometryException, ValueError):
    """The given values are not collinear."""


class NotCoplanar(GeometryException, ValueError):
    """The given values are not coplanar."""


class IncidenceError(GeometryException, ValueError):
    """The given objects were not incident to each other."""


class LinearDependenceError(GeometryException, ValueError):
    """The given values were linearly dependent, making the computation impossible."""


class NotReducible(GeometryException, ValueError):
    """The given geometric object is not reducible."""
