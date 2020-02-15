from itertools import combinations

import numpy as np

from .base import EQ_TOL_ABS, EQ_TOL_REL, Tensor
from .utils import distinct, is_multiple
from .point import Line, Plane, Point, infty_hyperplane
from .transformation import rotation, translation
from .operators import dist, angle, harmonic_set, crossratio
from .exceptions import NotCoplanar, LinearDependenceError


class Polytope(Tensor):
    """A class representing polytopes in arbitrary dimension. A (n+1)-polytope is a collection of n-polytopes that
    have some (n-1)-polytopes in common, where 3-polytopes are polyhedra, 2-polytopes are polygons and 1-polytopes are
    line segments.

    Parameters
    ----------
    *args
        The polytopes defining the facets ot the polytope.

    Attributes
    ----------
    array : numpy.ndarray
        The underlying numpy array.

    """

    def __init__(self, *args):
        if len(args) > 1:
            args = tuple(a.array for a in args)
        super(Polytope, self).__init__(*args, covariant=[-1])

    def __apply__(self, transformation):
        result = self.copy()
        result.array = self.array.dot(transformation.array.T)
        return result

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, ", ".join(str(v) for v in self.vertices))

    @property
    def dim(self):
        """int: The dimension of the space that the polytope lives in."""
        return self.shape[-1] - 1

    @staticmethod
    def _normalize_array(array):
        array = array.T.astype(complex)
        isinf = np.isclose(array[-1], 0, atol=EQ_TOL_ABS)
        array[:, ~isinf] = array[:, ~isinf] / array[-1, ~isinf]
        array = np.real_if_close(array)
        return array.T

    @property
    def normalized_array(self):
        """numpy.ndarray: Array containing only normalized vertex coordinates."""
        return self._normalize_array(self.array)

    @property
    def vertices(self):
        """list of Point: The vertices of the polytope."""
        vertices = self.array.reshape(-1, self.shape[-1])
        return list(distinct(Point(x) for x in vertices))

    @property
    def facets(self):
        """list of Polytope: The facets of the polytope."""
        def poly(array):
            if array.ndim == 1:
                return Point(array)
            if array.ndim == 2:
                if len(array) == 2:
                    return Segment(array)
                if len(array) == 3:
                    return Triangle(array)
                elif len(array) == 4:
                    try:
                        return Rectangle(array)
                    except NotCoplanar:
                        return Polytope(array)
            try:
                return Polygon(array)
            except NotCoplanar:
                return Polytope(array)

        return [poly(x) for x in self.array]

    def __eq__(self, other):
        if isinstance(other, Polytope):

            if self.shape != other.shape:
                return False

            if self.rank > 2:
                # facets equal up to reordering
                return all(f in other.facets for f in self.facets) and all(f in self.facets for f in other.facets)

            return np.all(is_multiple(self.array, other.array, axis=-1, rtol=EQ_TOL_REL, atol=EQ_TOL_ABS))

        return super(Polytope, self).__eq__(other)

    def __add__(self, other):
        if isinstance(other, Point):
            return type(self)(*[f + other for f in self.facets])
        return super(Polytope, self).__add__(other)


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

    def __init__(self, *args):
        super(Segment, self).__init__(*args)
        self._line = Line(Point(self.array[0]), Point(self.array[1]))

    def __apply__(self, transformation):
        result = super(Segment, self).__apply__(transformation)
        result._line = Line(Point(result.array[0]), Point(result.array[1]))
        return result

    def contains(self, other, tol=1e-8):
        """Tests whether a point is contained in the segment.

        Parameters
        ----------
        other : Point
            The point to test.
        tol : float, optional
            The accepted tolerance.

        Returns
        -------
        bool
            True if the point is contained in the segment.

        """
        if not self._line.contains(other):
            return False

        p, q = self.vertices
        d = Point(q.normalized_array - p.normalized_array)

        m = self._line.basis_matrix
        p = Point(m.dot(p.array))
        q = Point(m.dot(q.array))
        d = Point(m.dot(d.array))
        other = Point(m.dot(other.array))

        cr = crossratio(d, p, q, other)

        return 0 <= cr + tol and cr <= 1 + tol

    def intersect(self, other):
        """Intersect the line segment with another object.

        Parameters
        ----------
        other : Line, Plane, Segment, Polygon or Polyhedron
            The object to intersect the line segment with.

        Returns
        -------
        list of Point
            The points of intersection.

        """
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

        if isinstance(other, (Polygon, Polyhedron)):
            return other.intersect(self)

    @property
    def midpoint(self):
        """Point: The midpoint of the segment."""
        l = self._line.meet(infty_hyperplane(self.dim))
        return harmonic_set(*self.vertices, l)

    @property
    def length(self):
        """float: The length of the segment."""
        return dist(*self.vertices)


class Simplex(Polytope):
    """Represents a simplex in any dimension, i.e. a k-polytope with k+1 vertices where k is the dimension.

    The simplex determined by k+1 points is given by the convex hull of these points.

    Parameters
    ----------
    *args
        The points that are the vertices of the simplex.

    """

    def __new__(cls, *args):
        if len(args) == 2:
            return Segment(*args)

        return super(Polytope, cls).__new__(cls)

    def __init__(self, *args):
        if len(args) > 3:
            args = [Simplex(*x) for x in combinations(args, len(args)-1)]
        super(Simplex, self).__init__(*args)

    @property
    def volume(self):
        """float: The volume of the simplex, calculated using the Cayley–Menger determinant."""
        points = np.concatenate([v.array.reshape((1, v.shape[0])) for v in self.vertices], axis=0)
        points = self._normalize_array(points)
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
        if all(isinstance(x, Segment) for x in args):
            args = (np.array([s.array[0] for s in args]),)
        super(Polygon, self).__init__(*args)
        self._plane = Plane(*self.vertices[:self.dim]) if self.dim > 2 else None

    def __apply__(self, transformation):
        result = super(Polygon, self).__apply__(transformation)
        if result.dim > 2:
            result._plane = Plane(*result.vertices[:result.dim])
        return result

    @property
    def vertices(self):
        return [Point(x) for x in self.array]

    @property
    def facets(self):
        segments = []
        for i in range(len(self.array)):
            segments.append(Segment(self.array[[i, (i + 1) % len(self.array)]]))
        return segments

    @property
    def edges(self):
        """list of Segment: The edges of the polygon."""
        return self.facets

    def contains(self, other):
        """Tests whether a point is contained in the polygon.

        Parameters
        ----------
        other : Point
            The point to test.

        Returns
        -------
        bool
            True if the point is contained in the polygon.

        """

        if self.dim > 2 and not self._plane.contains(other):
            return False

        if other.isinf:
            direction = Point([1] + [0]*self.dim)
            if direction == other:
                direction = Point([1, 1] + [0]*(self.dim-1))
        elif self.dim == 2:
            direction = Point([1, 0, 0])
        else:
            a = self._plane.array
            if np.isclose(a[0], 0, atol=EQ_TOL_ABS):
                direction = Point([1] + [0] * self.dim)
            else:
                norm = np.linalg.norm(a[:2])
                direction = Point([a[1]/norm, -a[0]/norm] + [0]*(self.dim-1))

        ray = Segment(other, direction)
        intersection_count = sum(len(s.intersect(ray)) for s in self.edges)

        return intersection_count % 2 == 1

    def intersect(self, other):
        """Intersect the polygon with another object.

        Parameters
        ----------
        other : Line or Segment
            The object to intersect the polygon with.

        Returns
        -------
        list of Point
            The points of intersection.

        """
        if self.dim > 2:

            if isinstance(other, Line):
                try:
                    p = self._plane.meet(other)
                except LinearDependenceError:
                    return []
                return [p] if self.contains(p) else []

            if isinstance(other, Segment):
                if self._plane.contains(other._line):
                    return []

                i = other.intersect(self._plane)
                return i if i and self.contains(i[0]) else []

        return list(distinct(x for f in self.edges for x in f.intersect(other)))

    def _normalized_projection(self):
        points = self.array

        if self.dim > 2:
            e = self._plane
            o = Point(*[0] * self.dim)
            if not e.contains(o):
                # use parallel hyperplane for projection to avoid rescaling
                e = e.parallel(o)
            m = e.basis_matrix
            points = points.dot(m.T)

        return self._normalize_array(points)

    @property
    def area(self):
        """float: The area of the polygon."""
        points = self._normalized_projection()
        a = sum(np.linalg.det(points[[0, i, i + 1]]) for i in range(1, points.shape[0] - 1))
        return 1/2 * abs(a)

    @property
    def centroid(self):
        """Point: The centroid (center of mass) of the polygon."""
        points = self.normalized_array
        centroids = [np.average(points[[0, i, i + 1], :-1], axis=0) for i in range(1, points.shape[0] - 1)]
        weights = [np.linalg.det(self._normalized_projection()[[0, i, i + 1]])/2 for i in range(1, points.shape[0] - 1)]
        return Point(*np.average(centroids, weights=weights, axis=0))

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
    n : int
        The number of vertices of the regular polygon.
    axis : Point, optional
        If constructed in higher-dimensional spaces, an axis vector is required to orient the polygon.

    """

    def __init__(self, center, radius, n, axis=None):
        if axis is None:
            p = Point(1, 0)
        else:
            e = Plane(np.append(axis.array[:-1], [0]))
            p = Point(*e.basis_matrix[0, :-1])

        vertex = center + radius*p

        vertices = []
        for i in range(n):
            t = rotation(2*np.pi*i / n, axis=axis)
            t = translation(center) * t * translation(-center)
            vertices.append(t*vertex)

        super(RegularPolygon, self).__init__(*vertices)

    @property
    def radius(self):
        """float: The Circumradius of the regular polygon."""
        return dist(self.center, self.vertices[0])

    @property
    def center(self):
        """Point: The center of the polygon."""
        return Point(*np.sum(self.normalized_array[:, :-1], axis=0))

    @property
    def inradius(self):
        """float: The inradius of the regular polygon."""
        return dist(self.center, self.edges[0].midpoint)


class Triangle(Polygon, Simplex):
    """A class representing triangles.

    Parameters
    ----------
    a : Point
    b : Point
    c : Point

    """
    pass


class Rectangle(Polygon):
    """A class representing rectangles.

    Parameters
    ----------
    a : Point
    b : Point
    c : Point
    d : Point

    """
    pass


class Polyhedron(Polytope):
    """A class representing polyhedra (3-polytopes).

    """

    @property
    def faces(self):
        """list of Polygon: The faces of the polyhedron."""
        return self.facets

    @property
    def edges(self):
        """list of Segment: The edges of the polyhedron."""
        return list(distinct(s for f in self.faces for s in f.edges))

    @property
    def area(self):
        """float: The surface area of the polyhedron."""
        return sum(s.area for s in self.faces)

    def intersect(self, other):
        """Intersect the polyhedron with another object.

        Parameters
        ----------
        other : Line or Segment
            The object to intersect the polyhedron with.

        Returns
        -------
        list of Point
            The points of intersection.

        """
        return list(distinct(x for f in self.faces for x in f.intersect(other)))


class Cuboid(Polyhedron):
    """A class that can be used to construct a cuboid/box or a cube.

    Parameters
    ----------
    a : Point
        The base point of the cuboid.
    b : Point
        The vertex that determines the first direction of the edges.
    c : Point
        The vertex that determines the second direction of the edges.
    d : Point
        The vertex that determines the third direction of the edges.

    """

    def __init__(self, a, b, c, d):
        x, y, z = b-a, c-a, d-a
        yz = Rectangle(a, a + z, a + y + z, a + y)
        xz = Rectangle(a, a + x, a + x + z, a + z)
        xy = Rectangle(a, a + x, a + x + y, a + y)
        super(Cuboid, self).__init__(yz, xz, xy, yz + x, xz + y, xy + z)
