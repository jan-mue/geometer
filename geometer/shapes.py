from itertools import combinations

import numpy as np

from .utils import distinct
from .point import Line, Plane, Point, join, infty_hyperplane
from .operators import dist, angle, harmonic_set
from .exceptions import NotCoplanar, LinearDependenceError


class Polytope:
    """A class representing polytopes in arbitrary dimension. A (n+1)-polytope is a collection of n-polytopes that
    have some (n-1)-polytopes in common, where 3-polytopes are polyhedra, 2-polytopes are polygons and 1-polytopes are
    line segments.

    Parameters
    ----------
    *args
        The polytopes defining the facets ot the polytope.

    """

    def __init__(self, *args):
        self._facets = list(args)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, ", ".join(str(v) for v in self.vertices))

    @property
    def vertices(self):
        """list of Point: The vertices of the polytope."""
        return list(distinct(v for f in self.facets for v in f.vertices))

    @property
    def dim(self):
        """int: The dimension of the space that the polytope lives in."""
        return self.facets[0].dim

    @property
    def facets(self):
        """list of Polytope: The facets of the polytope."""
        return self._facets

    def intersect(self, other):
        """Intersect the polytope with another object.

        Parameters
        ----------
        other : Line, Plane, Quadric or Polytope

        Returns
        -------
        Polytope, Quadric or list of Point
            The points of intersection.

        """
        if isinstance(other, Plane):
            facets = [f.intersect(other) for f in self.facets]
            return Polytope(*[f for f in facets if f])

        return list(distinct(pt for f in self.facets for pt in f.intersect(other) if f != other))

    def __eq__(self, other):
        if isinstance(other, Polytope):
            # facets equal up to reordering
            return np.all(f in other.facets for f in self.facets) and np.all(f in self.facets for f in other.facets)
        return NotImplemented

    def __add__(self, other):
        return type(self)(*[s + other for s in self.facets])


class Segment(Polytope):
    """Represents a line segment in an arbitrary projective space.

    As a (real) projective line is homeomorphic to a circle, there are two line segments that connect two points. An
    instance of this class will represent the finite segment connecting the two points, if there is one, and the segment
    in the direction of the infinite point otherwise (identifying only scalar multiples by positive scalar factors).
    When both points are at infinity, the points will be considered in the oriented projective space to define the
    segment between them.

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
        return self._facets

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


class Simplex(Polytope):
    """Represents a simplex in any dimension, i.e. a k-polytope with k+1 vertices where k is the dimension.

    The simplex determined by k+1 points is given by the convex hull of these points.

    Parameters
    ----------
    *args
        The points that are the vertices of the simplex.

    """

    def __new__(cls, *args, **kwargs):
        if len(args) == 2 and all(isinstance(x, Point) for x in args):
            return Segment(*args)

        return super(Polytope, cls).__new__(cls)

    def __init__(self, *args):
        if all(isinstance(x, Point) for x in args):
            args = [Simplex(*x) for x in combinations(args, len(args)-1)]

        super(Simplex, self).__init__(*args)

    def volume(self):
        """Calculates the volume of the simplex using the Cayley–Menger determinant.

        For a 2-dimensional simplex this will be the same as the area of a triangle.

        Returns
        -------
        float
            The volume of the simplex.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Cayley%E2%80%93Menger_determinant

        """
        points = np.concatenate([v.array.reshape((1, v.array.shape[0])) for v in self.vertices], axis=0)
        points = (points.T / points[:, -1]).T
        n, k = points.shape

        if n == k:
            return 1 / np.math.factorial(n-1) * abs(np.linalg.det(points))

        indices = np.triu_indices(n)
        distances = points[indices[0]] - points[indices[1]]
        distances = np.sum(distances**2, axis=1)
        m = np.zeros((n+1, n+1), dtype=distances.dtype)
        m[indices] = distances
        m += m.T
        m[-1, :-1] = 1
        m[:-1, -1] = 1

        return np.sqrt((-1)**n/(np.math.factorial(n-1)**2 * 2**(n-1)) * np.linalg.det(m))


class Polygon(Polytope):
    """A flat polygon with vertices in any dimension.

    Parameters
    ----------
    *args
        The coplanar points that are the vertices of the polygon. They will be connected sequentially by line segments.

    """

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

    @property
    def edges(self):
        return self.facets

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

        # TODO: ignore intersection with vertices

        return len(super(Polygon, self).intersect(s)) % 2 == 1

    def intersect(self, other):
        if self.dim > 2:

            if isinstance(other, Line):
                if self._plane.contains(other):
                    pts = super(Polygon, self).intersect(other)
                    return Segment(*pts) if pts else []

                p = self._plane.meet(other)
                return [p] if self.contains(p) else []

            if isinstance(other, Plane):
                if self._plane == other:
                    return self

                l = self._plane.meet(other)
                return self.intersect(l)

            # TODO: handle intersection of coplanar polygons correctly
            if isinstance(other, Polygon) and self._plane != other._plane:
                l = self._plane.meet(other._plane)
                i1 = self.intersect(l)
                i2 = other.intersect(l)
                pts = [p for p in i1 + i2 if self.contains(p) and other.contains(p)]
                return [Segment(pts[i], pts[i+1]) for i in range(0, len(pts), 2)]

            if isinstance(other, Segment) and not self._plane.contains(other):
                i = other.intersect(self._plane)
                return i if i and self.contains(i[0]) else []

            if isinstance(other, Polytope) and not isinstance(other, Segment):
                return other.intersect(self)

        return super(Polygon, self).intersect(other)

    def area(self):
        """Calculates the area of the polygon.

        Returns
        -------
        float
            The area of the polygon.

        """
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
        """list of float: The interior angles of the polygon."""
        result = []
        a = self.edges[-1]
        for b in self.edges:
            result.append(angle(a.vertices[1], a.vertices[0], b.vertices[1]))
            a = b

        return result


class RegularPolygon(Polygon):
    """A class that can be used to construct regular polygon from a radius and a center point.

    Parameters
    ----------
    center : Point
        The center of the polygon.
    radius : float
        The distance from the center to the vertices of the polygon.
    axis : Point, optional
        If constructed in higher-dimensional spaces, an axis vector is required to orient the polygon.

    """

    def __init__(self, center, radius, n, axis=None):
        self.center = center
        if axis is None:
            p = Point(1, 0)
        else:
            e = Plane(np.append(axis.array[:-1], [0]))
            p = Point(*e.basis_matrix[0, :-1])
        vertex = center + radius*p

        from .transformation import rotation
        super(RegularPolygon, self).__init__(*[rotation(2*i*np.pi/n, axis=axis)*vertex for i in range(n)])


class Triangle(Polygon, Simplex):
    """A class representing triangles.

    Parameters
    ----------
    a : Point
    b : Point
    c : Point

    """

    def __init__(self, a, b, c):
        super(Triangle, self).__init__(a, b, c)


class Rectangle(Polygon):
    """A class representing rectangles.

    Parameters
    ----------
    a : Point
    b : Point
    c : Point
    d : Point

    """

    def __init__(self, a, b, c, d):
        super(Rectangle, self).__init__(a, b, c, d)


class Polyhedron(Polytope):
    """A class representing polyhedra (3-polytopes).

    """

    @property
    def faces(self):
        return self.facets

    def area(self):
        """Calculates the surface area of the polyhedron.

        Returns
        -------
        float
            The surface area of the polyhedron.

        """
        return sum(s.area() for s in self.faces)


class Cuboid(Polyhedron):
    """A class that can be used to construct a cuboid/box or a cube.

    Parameters
    ----------
    a : Point
        The base point of the cuboid.
    b : Point
        The vertex that determines the first direction of the edges.
    b : Point
        The vertex that determines the second direction of the edges.
    b : Point
        The vertex that determines the third direction of the edges.

    """

    def __init__(self, a, b, c, d):
        x, y, z = b-a, c-a, d-a
        yz = Rectangle(a, a + z, a + y + z, a + y)
        xz = Rectangle(a, a + x, a + x + z, a + z)
        xy = Rectangle(a, a + x, a + x + y, a + y)
        super(Cuboid, self).__init__(yz, xz, xy, yz + x, xz + y, xy + z)
