import numpy as np
from .point import Line, Point
from .base import GeometryObject


class Segment(GeometryObject):

    def plot(self):
        pass

    def __init__(self, p: Point, q: Point):
        self._points = (p, q)
        self._line = Line(p, q)

    def contains(self, pt: Point):
        if not self._line.contains(pt):
            return False

        p, q = self._points
        d = (pt-p).array[:-1].dot((q-p).array[:-1])

        if d < 0 or d > (q-p).array[:-1].dot((q-p).array[:-1]):
            return False

        return True

    def intersect(self, other):
        if isinstance(other, Line):
            pt = other.meet(self._line)
            return pt if self.contains(pt) else None
        if isinstance(other, Segment):
            pt = other.intersect(self._line)
            return pt if pt is None or self.contains(pt) else None


class Polygon(GeometryObject):

    def __init__(self, *args):
        self._segments = []
        self._vertices = list(args)
        a = None
        for b in args:
            if a is not None:
                self._segments.append(Segment(a, b))
            a = b
        self._segments.append(Segment(a, args[0]))

    @property
    def vertices(self):
        return self._vertices

    @property
    def angles(self):
        # TODO: implement this
        return {}

    def convex_hull(self):
        # TODO: find efficient algorithm
        return self

    def triangulate(self):
        # TODO: implement for general polynomial
        return []

    def area(self):
        return sum(t.area() for t in self.triangulate())

    def plot(self):
        for s in self._segments:
            s.plot()

    def contains(self, pt: Point):
        for s in self._segments:
            if s.contains(pt):
                return True
        return False

    def intersect(self, other):
        if isinstance(other, (Line, Segment)):
            result = []
            for s in self._segments:
                i = s.intersect(other)
                if i is not None and i not in result:
                    result.append(i)
            return result
        if isinstance(other, Polygon):
            return sum(self.intersect(s) for s in other._segments)


class RegularPolygon(Polygon):

    def __init__(self, center, radius, n):
        self.center = center
        # TODO: implement this
        super(RegularPolygon, self).__init__()


class Triangle(Polygon):

    def __init__(self, a, b, c):
        super(Triangle, self).__init__(a, b, c)

    def area(self):
        return 1/2 * np.linalg.det([v.normalized().array for v in self.vertices])


class Rectangle(Polygon):

    def __init__(self, a, b, c, d):
        super(Rectangle, self).__init__(a, b, c, d)
