from abc import ABC, abstractmethod
import numpy as np


class GeometryObject(ABC):

    @abstractmethod
    def plot(self):
        pass

    @abstractmethod
    def intersect(self, other):
        pass


class ProjectiveElement(GeometryObject, ABC):

    def __init__(self, *args):
        if len(args) == 1:
            self.array = np.atleast_1d(args[0])
        else:
            self.array = np.array([*args])

    def __eq__(self, other):
        # By Cauchy-Schwarz |(x,y)| = ||x||*||y|| iff x = cy
        a = self.array.flatten()
        b = other.array.flatten()
        return np.isclose(float(a.dot(b)**2), float(a.dot(a)*b.dot(b)))

    def __len__(self):
        return np.product(self.array.shape)
