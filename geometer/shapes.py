import numpy as np
import scipy.spatial
from .base import Shape
from .point import Line, Plane, Point, join, infty_hyperplane
from .operators import dist, angle, harmonic_set
from .exceptions import NotCoplanar


class Segment(Shape):
    """Represents a line segment in an arbitrary projective space.

    The order of the arguments matters! As a (real) projective line is homeomorphic to a circle, there are two line
    segments that connect two points.

    Parameters
    ----------
    p : Point
        The start of the line segment.
    q : Point
        The end point of the segment.

    """

    def __init__(self, p: Point, q: Point):
        self._points = (p, q)
        self._line = Line(p, q)

    @property
    def vertices(self):
        return list(self._points)

    def contains(self, pt: Point):
        if not self._line.contains(pt):
            return False

        p, q = self._points
        d = (pt - p).array[:-1].dot((q - p).array[:-1])

        if d < 0 or d > (q - p).array[:-1].dot((q - p).array[:-1]):
            return False

        return True

    def intersect(self, other):
        if isinstance(other, (Line, Plane)):
            pt = other.meet(self._line)
            return [pt] if self.contains(pt) else []
        if isinstance(other, Segment):
            i = other.intersect(self._line)
            return i if i and self.contains(i[0]) else []
        return []

    def complement(self):
        return Segment(*reversed(self._points))

    def midpoint(self):
        l = self._line.meet(infty_hyperplane(self.dim))
        return harmonic_set(*self._points, l)

    def length(self):
        return dist(*self._points)

    def __eq__(self, other):
        return self.vertices == other.vertices

    def __add__(self, other):
        return Segment(*[p + other for p in self._points])


class Polytope(Shape):
    """A class representing polytopes in arbitrary dimension.

    When a list of points is passed to the init method, the resulting polytope will be the convex hull of the points.

    Parameters
    ----------
    *args
        The polygons defining the faces ot the polytope or a list of points.

    """

    def __init__(self, *args):

        if len(args) < 3:
            raise ValueError("Expected at least 3 arguments, got {}.".format(str(len(args))))

        if all(isinstance(x, Point) for x in args):
            self._sides = self._convex_hull_from_vertices(args)
        else:
            self._sides = list(args)

    @property
    def vertices(self):
        result = []
        for s in self._sides:
            for v in s.vertices:
                if v not in result:
                    result.append(v)
        return result

    @property
    def sides(self):
        return self._sides

    @staticmethod
    def _sum_lists(iterable):
        result = []
        for l in iterable:
            for x in l:
                if x not in result:
                    result.append(x)
        return result

    def intersect(self, other):
        if isinstance(other, Polytope):
            return self._sum_lists(self.intersect(s) for s in other.sides)
        return self._sum_lists(s.intersect(other) for s in self.sides)

    @staticmethod
    def _convex_hull_from_vertices(vertices):
        hull = scipy.spatial.ConvexHull([p.normalized_array[:-1] for p in vertices])
        return [_simplex(*[vertices[i] for i in a]) for a in hull.simplices]

    def convex_hull(self):
        """Calculates the convex hull of the vertices of the polytope.

        In a projective space a convex set is defined as follows:

            A set S is convex if for any two points of S, exactly one of the segments determined by these points
            is contained in S.

        Returns
        -------
        Polytope
            The convex hull of the vertices.

        """
        sides = self._convex_hull_from_vertices(self.vertices)
        return Polytope(*sides)

    def triangulate(self):
        points = self.vertices
        a = np.concatenate([p.array.reshape((p.array.shape[0], 1)) for p in points], axis=1)
        tri = scipy.spatial.Delaunay((a[:-1] / a[-1]).T)
        return [_simplex(*[points[i] for i in a]) for a in tri.simplices]

    def is_convex(self):
        # TODO: implement this
        return True

    def volume(self):
        return sum(t.volume() for t in self.triangulate())

    def area(self):
        return sum(s.area() for s in self.sides)

    def __eq__(self, other):
        return self.sides == other.sides

    def __add__(self, other):
        return type(self)(*[s + other for s in self.sides])


class Simplex(Polytope):

    def __init__(self, *args):
        super(Simplex, self).__init__(*args)
        if len(self.vertices) != self.dim + 1:
            raise ValueError("Expected {} vertices, got {}.".format(str(self.dim + 1), str(len(self.vertices))))

    def volume(self):
        points = np.concatenate([v.array.reshape((v.array.shape[0], 1)) for v in self.vertices], axis=1)
        return 1 / np.math.factorial(self.dim) * abs(np.linalg.det(points / points[-1]))


def _simplex(*args):
    if len(args) == 2:
        return Segment(*args)
    if len(args) == 3:
        return Triangle(*args)
    return Simplex(*args)


class Polygon(Polytope):

    def __init__(self, *args):
        super(Polygon, self).__init__(*args)
        self._plane = None
        if self.dim > 2:
            self._plane = join(*self.vertices[:self.dim])
            for s in args[self.dim:]:
                if not self._plane.contains(s):
                    raise NotCoplanar("The vertices of a polygon must be coplanar!")

    def contains(self, pt: Point):
        # TODO: fix for dim > 2
        for s in self.sides:
            if np.any(s._line.array.dot(pt.array) > 0):
                return False
        return True

    def intersect(self, other):
        if self.dim > 2:
            if isinstance(other, Line):
                p = self._plane.meet(other)
                return [p] if self.contains(p) else []
            if isinstance(other, Segment):
                i = other.intersect(self._plane)
                return i if i and self.contains(i[0]) else []
        return super(Polygon, self).intersect(other)

    def triangulate(self):
        if self.dim == 2:
            return super(Polygon, self).triangulate()
        vertices = self.vertices
        m = self._plane.basis_matrix
        points = np.concatenate([v.array.reshape((m.shape[1], 1)) for v in vertices], axis=1)
        points = m.dot(points)
        points = points / points[-1]
        tri = scipy.spatial.Delaunay(points.T[:, :-1])
        return [Triangle(*[vertices[i] for i in a]) for a in tri.simplices]

    def area(self):
        return sum(t.area() for t in self.triangulate())

    @property
    def angles(self):
        result = []
        a = self.sides[-1]
        for b in self.sides:
            result.append(angle(a.vertices[1], a.vertices[0], b.vertices[1]))
            a = b

        return result


class RegularPolygon(Polygon):

    def __init__(self, center, radius, n):
        self.center = center
        # TODO: implement this
        super(RegularPolygon, self).__init__()


class Triangle(Polygon):

    def __init__(self, *args):
        if len(args) != 3 or any(not isinstance(x, Point) for x in args):
            super(Triangle, self).__init__(*args)
        else:
            a, b, c = args
            super(Triangle, self).__init__(Segment(a, b), Segment(b, c), Segment(c, a))

    def area(self):
        points = np.concatenate([v.array.reshape((v.array.shape[0], 1)) for v in self.vertices], axis=1)
        if self.dim == 2:
            return 1 / 2 * abs(np.linalg.det(points / points[-1]))
        e = self._plane
        o = Point(*[0]*self.dim)
        if not e.contains(o):
            # use parallel hyperplane for projection to avoid rescaling
            e = e.parallel(o)
        m = e.basis_matrix
        points = m.dot(points)
        return 1 / 2 * abs(np.linalg.det(points / points[-1]))


class Rectangle(Polygon):

    def __init__(self, *args):
        if len(args) != 4 or any(not isinstance(x, Point) for x in args):
            super(Rectangle, self).__init__(*args)
        else:
            a, b, c, d = args
            super(Rectangle, self).__init__(Segment(a, b), Segment(b, c), Segment(c, d), Segment(d, a))


class Cube(Polytope):

    def __init__(self, *args):
        if len(args) != 4 or any(not isinstance(x, Point) for x in args):
            super(Cube, self).__init__(*args)
        else:
            a, b, c, d = args
            x, y, z = b-a, c-a, d-a
            yz = Rectangle(a, a + z, a + y + z, a + y)
            xz = Rectangle(a, a + x, a + x + z, a + z)
            xy = Rectangle(a, a + x, a + x + y, a + y)
            super(Cube, self).__init__(yz, xz, xy, yz + x, xz + y, xy + z)
