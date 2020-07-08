from itertools import combinations

import numpy as np

from .base import EQ_TOL_ABS, EQ_TOL_REL, Tensor
from .utils import distinct, is_multiple, det
from .point import Line, Plane, Point, infty_hyperplane
from .transformation import rotation, translation
from .operators import dist, angle, harmonic_set, crossratio
from .exceptions import NotCoplanar, LinearDependenceError


class Polytope(Tensor):
    """A class representing polytopes in arbitrary dimension. A (n+1)-polytope is a collection of n-polytopes that
    have some (n-1)-polytopes in common, where 3-polytopes are polyhedra, 2-polytopes are polygons and 1-polytopes are
    line segments.

    The polytope is stored as a multidimensional numpy array. Hence, all facets of the polytope must have the same
    number of vertices and facets.

    Parameters
    ----------
    *args
        The polytopes defining the facets ot the polytope.
    **kwargs
        Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    Attributes
    ----------
    array : numpy.ndarray
        The underlying numpy array.

    """

    def __init__(self, *args, **kwargs):
        if len(args) > 1:
            args = tuple(a.array for a in args)
        super(Polytope, self).__init__(*args, **kwargs, covariant=[-1])

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
        isinf = np.isclose(array[..., -1], 0, atol=EQ_TOL_ABS)
        result = array.astype(np.complex128)
        result[~isinf] /= array[~isinf, -1, None]
        return np.real_if_close(result)

    @property
    def normalized_array(self):
        """numpy.ndarray: Array containing only normalized vertex coordinates."""
        return self._normalize_array(self.array)

    @property
    def vertices(self):
        """list of Point: The vertices of the polytope."""
        vertices = self.array.reshape(-1, self.shape[-1])
        return list(distinct(Point(x, copy=False) for x in vertices))

    @property
    def facets(self):
        """list of Polytope: The facets of the polytope."""
        def poly(array):
            if array.ndim == 1:
                return Point(array, copy=False)

            if array.ndim == 2:
                if len(array) == 2:
                    return Segment(array, copy=False)
                if len(array) == 3:
                    return Triangle(array, copy=False)
                elif len(array) == 4:
                    try:
                        return Rectangle(array, copy=False)
                    except NotCoplanar:
                        return Polytope(array, copy=False)

            if array.ndim == 3:
                return Polyhedron(array, copy=False)

            try:
                return Polygon(array, copy=False)
            except NotCoplanar:
                return Polytope(array, copy=False)

        return [poly(x) for x in self.array]

    @property
    def _edges(self):
        v1 = self.array
        v2 = np.roll(v1, -1, axis=-2)
        return np.stack([v1, v2], axis=-2)

    def __eq__(self, other):
        if isinstance(other, Polytope):

            if self.shape != other.shape:
                return False

            if self.rank > 2:
                # facets equal up to reordering
                facets1 = self.facets
                facets2 = other.facets
                return all(f in facets2 for f in facets1) and all(f in facets1 for f in facets2)

            # edges equal up to circular reordering
            edges1 = self._edges
            edges2 = other._edges

            for i in range(self.shape[0]):
                if np.all(is_multiple(edges1, np.roll(edges2, i, axis=0), axis=-1, rtol=EQ_TOL_REL, atol=EQ_TOL_ABS)):
                    return True

            return False

        return super(Polytope, self).__eq__(other)

    def __add__(self, other):
        if not isinstance(other, Point):
            return super(Polytope, self).__add__(other)
        return translation(other).apply(self)

    def __sub__(self, other):
        return self + (-other)


class Segment(Polytope):
    """Represents a line segment in an arbitrary projective space.

    Segments with one point at infinity represent rays/half-lines in a traditional sense.

    Parameters
    ----------
    *args
        The start and endpoint of the line segment, either as two Point objects or a single coordinate array.
    **kwargs
        Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    """

    def __init__(self, *args, **kwargs):
        super(Segment, self).__init__(*args, **kwargs)
        self._line = Line(Point(self.array[0], copy=False), Point(self.array[1], copy=False))

    def __apply__(self, transformation):
        result = super(Segment, self).__apply__(transformation)
        result._line = Line(Point(result.array[0]), Point(result.array[1]))
        return result

    @property
    def _edges(self):
        return self.array

    def contains(self, other, tol=EQ_TOL_ABS):
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

        m = self.array
        arr = m.dot(m.T)

        p = Point(arr[:, 0], copy=False)
        q = Point(arr[:, 1], copy=False)
        d = Point(arr[:, 1] - arr[:, 0], copy=False)
        other = Point(m.dot(other.array), copy=False)

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
    **kwargs
        Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    """

    def __new__(cls, *args, **kwargs):
        if len(args) == 2:
            return Segment(*args, **kwargs)

        return super(Polytope, cls).__new__(cls)

    def __init__(self, *args, **kwargs):
        if len(args) > 3:
            args = [Simplex(*x) for x in combinations(args, len(args)-1)]
        super(Simplex, self).__init__(*args, **kwargs)

    @property
    def volume(self):
        """float: The volume of the simplex, calculated using the Cayley–Menger determinant."""
        points = np.concatenate([v.array.reshape((1, v.shape[0])) for v in self.vertices], axis=0)
        points = self._normalize_array(points)
        n, k = points.shape

        if n == k:
            return 1 / np.math.factorial(n-1) * abs(det(points))

        indices = np.triu_indices(n)
        distances = points[indices[0]] - points[indices[1]]
        distances = np.sum(distances**2, axis=1)
        m = np.zeros((n+1, n+1), dtype=distances.dtype)
        m[indices] = distances
        m += m.T
        m[-1, :-1] = 1
        m[:-1, -1] = 1

        return np.sqrt((-1)**n/(np.math.factorial(n-1)**2 * 2**(n-1)) * det(m))


class Polygon(Polytope):
    """A flat polygon with vertices in any dimension.

    Parameters
    ----------
    *args
        The coplanar points that are the vertices of the polygon. They will be connected sequentially by line segments.
    **kwargs
        Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    """

    def __init__(self, *args, **kwargs):
        if all(isinstance(x, Segment) for x in args):
            args = (np.array([s.array[0] for s in args]),)
            kwargs['copy'] = False
        super(Polygon, self).__init__(*args, **kwargs)
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
        return [Segment(x, copy=False) for x in self._edges]

    @property
    def edges(self):
        """list of Segment: The edges of the polygon."""
        return self.facets

    def contains(self, other):
        """Tests whether a point is contained in the polygon.

        Points on an edge of the polygon are considered True.

        Parameters
        ----------
        other : Point
            The point to test.

        Returns
        -------
        bool
            True if the point is contained in the polygon.

        References
        ----------
        .. [1] http://paulbourke.net/geometry/polygonmesh/#insidepoly

        """

        if self.dim > 2:
            if not self._plane.contains(other):
                return False

            if self._plane == infty_hyperplane(self.dim):
                other = Point(other.array[:-1], copy=False)
                return Polygon(self.array[:, :-1], copy=False).contains(other)

            # remove coordinates with the largest absolute value in the normal vector
            i = np.argsort(np.abs(self._plane.array[:-1]))[2:]
            other = Point(np.delete(other.array, i), copy=False)

            edges = Polygon(np.delete(self.array, i, axis=-1), copy=False).edges

        else:
            edges = self.edges

        if any(s.contains(other) for s in edges):
            return True

        direction = [other.array[1], -other.array[0], 0] if other.isinf else [1, 0, 0]
        ray = Segment([other.array, direction])

        intersection_count = 0
        for s in edges:

            # ignore edges along the ray
            if s._line == ray._line:
                continue

            v1, v2 = s.array
            for p in ray.intersect(s):
                # ignore intersections of downward edges that end on the ray
                if is_multiple(p.array, v1, rtol=EQ_TOL_REL, atol=EQ_TOL_ABS) and v1[1] <= v2[1]:
                    continue
                if is_multiple(p.array, v2, rtol=EQ_TOL_REL, atol=EQ_TOL_ABS) and v2[1] <= v1[1]:
                    continue

                intersection_count += 1

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
        a = sum(det(points[[0, i, i + 1]]) for i in range(1, points.shape[0] - 1))
        return 1/2 * abs(a)

    @property
    def centroid(self):
        """Point: The centroid (center of mass) of the polygon."""
        points = self.normalized_array
        centroids = [np.average(points[[0, i, i + 1], :-1], axis=0) for i in range(1, points.shape[0] - 1)]
        weights = [det(self._normalized_projection()[[0, i, i + 1]])/2 for i in range(1, points.shape[0] - 1)]
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
    **kwargs
        Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    """

    def __init__(self, center, radius, n, axis=None, **kwargs):
        if axis is None:
            p = Point(1, 0)
        else:
            e = Plane(np.append(axis.array[:-1], [0]), copy=False)
            p = Point(*e.basis_matrix[0, :-1], copy=False)

        vertex = center + radius*p

        vertices = []
        for i in range(n):
            t = rotation(2*np.pi*i / n, axis=axis)
            t = translation(center) * t * translation(-center)
            vertices.append(t*vertex)

        super(RegularPolygon, self).__init__(*vertices, **kwargs)

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
        result = self._edges
        return list(distinct(Segment(result[idx], copy=False) for idx in np.ndindex(self.shape[:2])))

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
    **kwargs
        Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    """

    def __init__(self, a, b, c, d, **kwargs):
        x, y, z = b-a, c-a, d-a
        yz = Rectangle(a, a + z, a + y + z, a + y)
        xz = Rectangle(a, a + x, a + x + z, a + z)
        xy = Rectangle(a, a + x, a + x + y, a + y)
        super(Cuboid, self).__init__(yz, xz, xy, yz + x, xz + y, xy + z, **kwargs)
