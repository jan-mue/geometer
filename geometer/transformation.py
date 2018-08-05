import numpy as np
from .point import Point, Line
from .curve import Conic


def rotation(angle:float):
    return Transformation(np.array([[np.cos(angle), - np.sin(angle), 0],
                                    [np.sin(angle), np.cos(angle), 0],
                                    [0, 0, 1]]))

def translation(*coordinates):
    m = np.eye(len(coordinates) + 1)
    m[:-1, -1] = [*coordinates]
    return Transformation(m)


class Transformation:

    def __init__(self, array):
        self.array = np.array(array)

    @classmethod
    def from_points(cls, a, b, c, d):
        a1, a2 = a
        b1, b2 = b
        c1, c2 = c
        d1, d2 = d
        m1 = np.array([a2.array, b2.array, c2.array]).T.dot(np.diag(d2.array))
        m2 = np.array([a1.array, b1.array, c1.array]).T.dot(np.diag(d1.array))
        return Transformation(m1.dot(np.linalg.inv(m2)))

    def __mul__(self, other):
        if isinstance(other, Point):
            return Point(self.array.dot(other.array))
        if isinstance(other, Line):
            return Line(np.linalg.solve(self.array.T, other.array))
        if isinstance(other, Conic):
            m = np.linalg.inv(self.array)
            return Conic(m.T.dot(other.array).dot(m))
        if isinstance(other, Transformation):
            return Transformation(self.array.dot(other.array))
        raise NotImplementedError

    def inverse(self):
        return Transformation(np.linalg.inv(self.array))

    def __rmul__(self, other):
        return self*other
