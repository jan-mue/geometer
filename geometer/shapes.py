import numpy as np

from .base import Shape
from .point import Line, Plane, Point, join, infty_hyperplane
from .operators import dist, angle, harmonic_set
from .transformation import rotation
from .exceptions import NotCoplanar, LinearDependenceError


class Segment(Shape):
    """Represents a line segment in an arbitrary projective space.

    As a (real) projective line is homeomorphic to a circle, there are two line segments that connect two points. An
    instance of this class will represent the finite segment connecting the two points, if there is one and the segment
    in the direction of the infinite point otherwise (ignoring the identification of scalar multiples). When both points
    are at infinity, the points will be considered in the oriented projective space to define the segment between them.

    Segments with one point at infinity represent rays/half-lines in a traditional sense.

    Parameters
    ----------
    p : Point
        The start of the line segment.
    q : Point
        The end point of the segment.

    """

    def __init__(self, p, q):
        self._line = Line(p, q)
        super(Segment, self).__init__(p, q)

    @property
    def vertices(self):
        return self._sides

    def contains(self, other):
        """Tests whether a point is contained in the segment.

        Parameters
        ----------
        other : Point
            The point to test.

        Returns
        -------
        bool
            True if the point is contained in the segment.

        """
        if not self._line.contains(other):
            return False

        p, q = self.vertices

        pinf = np.isclose(p.array[-1], 0)
        qinf = np.isclose(q.array[-1], 0)

        if np.isclose(other.array[-1], 0) and not (pinf and qinf):
            return other == p or other == q

        if pinf and not qinf:
            direction = - p.array[:-1]
            d2 = np.inf

        elif qinf and not pinf:
            direction = q.array[:-1]
            d2 = np.inf

        else:
            direction = (q - p).array[:-1]
            d2 = direction.dot(direction)

        d1 = (other - p).array[:-1].dot(direction)

        return 0 <= d1 <= d2

    def intersect(self, other):
        if isinstance(other, (Line, Plane)):
            try:
                pt = other.meet(self._line)
            except (LinearDependenceError, NotCoplanar):
                return []
            return [pt] if self.contains(pt) else []

        if isinstance(other, Segment):
            if self._line == other._line:
                return []

            i = other.intersect(self._line)
            return i if i and self.contains(i[0]) else []

        if isinstance(other, Polytope):
            return other.intersect(self)

        # TODO: intersect quadric

        return NotImplemented

    def midpoint(self):
        """Returns the midpoint of the segment.

        Returns
        -------
        Point
            The midpoint of the segment.

        """
        l = self._line.meet(infty_hyperplane(self.dim))
        return harmonic_set(*self.vertices, l)

    def length(self):
        """Returns the length of the segment.

        Returns
        -------
        float
            The length of the segment.

        """
        return dist(*self.vertices)

    def __eq__(self, other):
        if isinstance(other, Segment):
            return self.vertices == other.vertices
        return NotImplemented

    def __add__(self, other):
        return Segment(*[p + other for p in self.vertices])


class Polytope(Shape):
    """A class representing polytopes in arbitrary dimension. A (n+1)-polytope is a collection of n-polytopes that
    have some (n-1)-polytopes in common, where 3-polytopes are polyhedra and 2-polytopes are polygons.

    When a list of points is passed to the init method, the resulting polytope will be the convex hull of the points.

    Parameters
    ----------
    *args
        The polygons defining the faces ot the polytope or a list of points.

    """

    def __new__(cls, *args, **kwargs):
        if len(args) == 2:
            return Segment(*args)

        return super(Polytope, cls).__new__(cls)

    def __init__(self, *args):

        if len(args) < 3:
            raise ValueError("Expected at least 3 arguments, got {}.".format(str(len(args))))

        super(Polytope, self).__init__(*args)

    @property
    def vertices(self):
        """list of Point: The vertices of the polytope."""
        return self._sum_lists(s.vertices for s in self.sides)

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
            return self._sum_lists(self.intersect(s) for s in other.sides if s != other)
        return self._sum_lists(s.intersect(other) for s in self.sides if s != other)

    def area(self):
        """Calculates the surface area of the polytope.

        Works only for 3-polytopes, i.e. polyhedra with polygons as sides.

        Returns
        -------
        float
            The surface area of the polytope.

        """
        return sum(s.area() for s in self.sides)

    def __eq__(self, other):
        if isinstance(other, Polytope):
            return self.sides == other.sides
        return NotImplemented

    def __add__(self, other):
        return type(self)(*[s + other for s in self.sides])


class Simplex(Polytope):

    def __new__(cls, *args):
        if len(args) == 3:
            return Triangle(*args)

        return super(Simplex, cls).__new__(cls, *args)

    def __init__(self, *args):
        super(Simplex, self).__init__(*args)
        if len(self.vertices) != self.dim + 1:
            raise ValueError("Expected {} vertices, got {}.".format(str(self.dim + 1), str(len(self.vertices))))

    def volume(self):
        """Calculates the volume of the simplex.

        Returns
        -------
        float
            The volume of the simplex.

        """
        points = np.concatenate([v.array.reshape((v.array.shape[0], 1)) for v in self.vertices], axis=1)
        return 1 / np.math.factorial(self.dim) * abs(np.linalg.det(points / points[-1]))


class Polygon(Polytope):

    def __init__(self, *args):
        segments = []
        if all(isinstance(x, Point) for x in args):
            for i in range(len(args)):
                segments.append(Segment(args[i], args[(i+1) % len(args)]))
            args = segments
        super(Polygon, self).__init__(*args)
        self._plane = None
        if self.dim > 2:
            self._plane = join(*self.vertices[:self.dim])
            for s in self.vertices[self.dim:]:
                if not self._plane.contains(s):
                    raise NotCoplanar("The vertices of a polygon must be coplanar!")

    def contains(self, other):
        """Tests whether a point or segment is contained in the polygon.

        Parameters
        ----------
        other : Point or Segment
            The object to test.

        Returns
        -------
        bool
            True if the supplied object is contained in the polygon.

        """
        if isinstance(other, Segment):
            return len(self.intersect(other)) == 0

        e = infty_hyperplane(self.dim)
        if e.contains(other):
            p = Point([0]*self.dim + [1])
        elif self.dim == 2:
            p = Point([1, 0, 0])
        else:
            p = self._plane.meet(e).base_point
        s = Segment(other, p)
        return len(super(Polygon, self).intersect(s)) % 2 == 1

    def intersect(self, other):
        if self.dim > 2:
            if isinstance(other, Polygon) and self._plane != other._plane:
                l = self._plane.meet(other._plane)
                i1 = self.intersect(l)
                i2 = other.intersect(l)
                pts = [p for p in i1 + i2 if self.contains(p) and other.contains(p)]
                return [Segment(pts[i], pts[i+1]) for i in range(0, len(pts), 2)]

            if isinstance(other, Line) and not self._plane.contains(other):
                p = self._plane.meet(other)
                return [p] if self.contains(p) else []

            if isinstance(other, Segment) and not self._plane.contains(other):
                i = other.intersect(self._plane)
                return i if i and self.contains(i[0]) else []

        return super(Polygon, self).intersect(other)

    def area(self):
        points = np.concatenate([v.array.reshape((v.array.shape[0], 1)) for v in self.vertices], axis=1)

        if self.dim > 2:
            e = self._plane
            o = Point(*[0] * self.dim)
            if not e.contains(o):
                # use parallel hyperplane for projection to avoid rescaling
                e = e.parallel(o)
            m = e.basis_matrix
            points = m.dot(points)

        points = (points / points[-1]).T
        a = sum(np.linalg.det(points[[0, i, i + 1]]) for i in range(1, points.shape[0] - 1))
        return 1/2 * abs(a)

    @property
    def angles(self):
        """list of float: The interior angles of the polytope."""
        result = []
        a = self.sides[-1]
        for b in self.sides:
            result.append(angle(a.vertices[1], a.vertices[0], b.vertices[1]))
            a = b

        return result


class RegularPolygon(Polygon):

    def __init__(self, center, radius, n, axis=None):
        self.center = center
        if axis is None:
            p = Point(1, 0)
        else:
            e = Plane(np.append(axis.array[:-1], [0]))
            p = Point(*e.basis_matrix[0, :-1])
        vertex = center + radius*p
        super(RegularPolygon, self).__init__(*[rotation(2*i*np.pi/n, axis=axis)*vertex for i in range(n)])


class Triangle(Polygon):

    def __init__(self, a, b, c):
        super(Triangle, self).__init__(a, b, c)


class Rectangle(Polygon):

    def __init__(self, a, b, c, d):
        super(Rectangle, self).__init__(a, b, c, d)


class Cube(Polytope):

    def __init__(self, a, b, c, d):
        x, y, z = b-a, c-a, d-a
        yz = Rectangle(a, a + z, a + y + z, a + y)
        xz = Rectangle(a, a + x, a + x + z, a + z)
        xy = Rectangle(a, a + x, a + x + y, a + y)
        super(Cube, self).__init__(yz, xz, xy, yz + x, xz + y, xy + z)
