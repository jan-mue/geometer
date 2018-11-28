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
        a = self.array.ravel()
        b = other.array.ravel()
        return np.isclose(float(np.abs(np.vdot(a, b)))**2, float(np.vdot(a, a)*np.vdot(b, b)))

    def __len__(self):
        return np.product(self.array.shape)
