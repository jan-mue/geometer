import numpy as np
from .point import Line, Point, infty, infty_plane
from .operators import angle, harmonic_set


class Segment:

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

    def midpoint(self):
        l = self._line.meet(infty if self._line.dim == 2 else infty_plane)
        return harmonic_set(*self._points, l)


class Polygon:

    def __init__(self, *args):

        if len(args) < 2:
            raise ValueError("A polygon needs at least 2 vertices.")

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
        if len(self._segments) < 2:
            return []

        result = []
        a = self._segments[-1]
        for b in self._segments:
            result.append(angle(a._points[1], a._points[0], b._points[1]))
            a = b

        return result

    def convex_hull(self):
        # TODO: find efficient algorithm
        return self

    def triangulate(self):
        # TODO: implement for general polynomial
        return []

    def area(self):
        return sum(t.area() for t in self.triangulate())

    def contains(self, pt: Point):
        for s in self._segments:
            if s._line.array.dot(pt.array) > 0:
                return False
        return True

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
