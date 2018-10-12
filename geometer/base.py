from abc import ABC, abstractmethod


class GeometryObject(ABC):

    @abstractmethod
    def plot(self):
        pass

    @abstractmethod
    def intersect(self, other):
        pass
