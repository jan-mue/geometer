

class GeometryException(Exception):
    """A general geometric error occurred"""


class TensorComputationError(GeometryException):
    """A tensor computation could not be executed"""


class NotCollinear(GeometryException, ValueError):
    """The given values are not collinear"""


class NotCoplanar(GeometryException, ValueError):
    """The given values are not coplanar"""


class IncidenceError(GeometryException, ValueError):
    """The given values were not incident to each other"""


class LinearDependenceError(GeometryException, ValueError):
    """The given values were linearly dependent, making the computation impossible"""
