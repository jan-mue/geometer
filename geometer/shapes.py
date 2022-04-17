import math
from itertools import combinations

import numpy as np

from .base import EQ_TOL_ABS, EQ_TOL_REL
from .exceptions import LinearDependenceError, NotCoplanar
from .operators import angle, dist, harmonic_set
from .point import Line, LineCollection, Plane, Point, PointCollection, infty_hyperplane, join, meet
from .transformation import rotation, translation
from .utils import det, distinct, is_multiple, matmul, matvec


class Polytope(PointCollection):
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
        super(Polytope, self).__init__(args[0] if len(args) == 1 else args, **kwargs)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, ", ".join(str(v) for v in self.vertices))

    @property
    def vertices(self):
        """list of Point: The vertices of the polytope."""
        return list(distinct(self.flat))

    @property
    def facets(self):
        """list of Polytope: The facets of the polytope."""
        return list(self)

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

    def __getitem__(self, index):
        result = super(Polytope, self).__getitem__(index)

        if not isinstance(result, PointCollection):
            return result

        if result.rank == 2:
            if len(result) == 2:
                return Segment(result, copy=False)
            if len(result) == 3:
                return Triangle(result, copy=False)
            if len(result) == 4:
                try:
                    return Rectangle(result, copy=False)
                except NotCoplanar:
                    return Polytope(result, copy=False)

            try:
                return Polygon(result, copy=False)
            except NotCoplanar:
                return Polytope(result, copy=False)

        if result.rank == 3:
            return Polyhedron(result)

        return Polytope(result, copy=False)


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
        result._line = Line(Point(result.array[0], copy=False), Point(result.array[1], copy=False))
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
        return SegmentCollection.contains(self, other, tol=tol)

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
            args = [Simplex(*x) for x in combinations(args, len(args) - 1)]
        super(Simplex, self).__init__(*args, **kwargs)

    @property
    def volume(self):
        """float: The volume of the simplex, calculated using the Cayley–Menger determinant."""
        points = np.concatenate([v.array.reshape((1, v.shape[0])) for v in self.vertices], axis=0)
        points = self._normalize_array(points)
        n, k = points.shape

        if n == k:
            return 1 / math.factorial(n - 1) * abs(det(points))

        indices = np.triu_indices(n)
        distances = points[indices[0]] - points[indices[1]]
        distances = np.sum(distances ** 2, axis=1)
        m = np.zeros((n + 1, n + 1), dtype=distances.dtype)
        m[indices] = distances
        m += m.T
        m[-1, :-1] = 1
        m[:-1, -1] = 1

        return np.sqrt((-1) ** n / (math.factorial(n - 1) ** 2 * 2 ** (n - 1)) * det(m))


class Polygon(Polytope):
    """A flat polygon with vertices in any dimension.

    The vertices of the polygon must be given either in clockwise or counterclockwise order.

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
            kwargs["copy"] = False
        super(Polygon, self).__init__(*args, **kwargs)
        if self.dim > 2:
            vertices = self.vertices
            self._plane = Plane(*vertices[: self.dim])
            if any(not self._plane.contains(v) for v in vertices[self.dim:]):
                raise NotCoplanar("The given vertices are not coplanar.")
        else:
            self._plane = None

    def __apply__(self, transformation):
        result = super(Polygon, self).__apply__(transformation)
        if result.dim > 2:
            result._plane = Plane(*result.vertices[: result.dim])
        return result

    @property
    def vertices(self):
        return [Point(x, copy=False) for x in self.array]

    @property
    def facets(self):
        return list(self.edges)

    @property
    def edges(self):
        """SegmentCollection: The edges of the polygon."""
        return SegmentCollection(self._edges, copy=False)

    def contains(self, other):
        """Tests whether a point is contained in the polygon.

        Points on an edge of the polygon are considered True.

        Parameters
        ----------
        other : Point or PointCollection
            The point to test.

        Returns
        -------
        array_like
            True if the point is contained in the polygon.

        References
        ----------
        .. [1] http://paulbourke.net/geometry/polygonmesh/#insidepoly

        """
        return PolygonCollection.contains(self, other)

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

            # TODO: support collections

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

        return list(distinct(self.edges.intersect(other)))

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
        return 1 / 2 * abs(a)

    @property
    def centroid(self):
        """Point: The centroid (center of mass) of the polygon."""
        points = self.normalized_array
        centroids = [np.average(points[[0, i, i + 1], :-1], axis=0) for i in range(1, points.shape[0] - 1)]
        weights = [det(self._normalized_projection()[[0, i, i + 1]]) / 2 for i in range(1, points.shape[0] - 1)]
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

        vertex = center + radius * p

        vertices = []
        for i in range(n):
            t = rotation(2 * np.pi * i / n, axis=axis)
            t = translation(center) * t * translation(-center)
            vertices.append(t * vertex)

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

    @property
    def circumcenter(self):
        """Point: The circumcenter of the triangle."""
        e1, e2, e3 = self.edges
        bisector1 = e1._line.perpendicular(e1.midpoint, plane=self._plane)
        bisector2 = e2._line.perpendicular(e2.midpoint, plane=self._plane)
        return bisector1.meet(bisector2)

    def contains(self, other):
        # faster algorithm using barycentric coordinates

        a, b, c, p = np.broadcast_arrays(*self.array, other.array)

        lambda1 = det(np.stack([p, b, c], axis=-2))
        lambda2 = det(np.stack([a, p, c], axis=-2))

        result = (lambda1 <= 0) == (lambda2 <= 0)

        if not np.any(result):
            return result

        lambda3 = det(np.stack([a, b, p], axis=-2))

        area = lambda1 + lambda2 + lambda3

        if np.isscalar(area):
            if area < 0:
                return lambda1 <= 0 and lambda3 <= 0
            return lambda1 >= 0 and lambda3 >= 0

        ind = area < 0
        result[ind] &= (lambda1[ind] <= 0) & (lambda3[ind] <= 0)
        result[~ind] &= (lambda1[~ind] >= 0) & (lambda3[~ind] >= 0)

        return result


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
    """A class representing polyhedra (3-polytopes)."""

    @property
    def faces(self):
        """PolygonCollection: The faces of the polyhedron."""
        return PolygonCollection(self.array, copy=False)

    @property
    def edges(self):
        """list of Segment: The edges of the polyhedron."""
        result = self._edges
        return list(distinct(Segment(result[idx], copy=False) for idx in np.ndindex(self.shape[:2])))

    @property
    def area(self):
        """float: The surface area of the polyhedron."""
        return np.sum(self.faces.area)

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
        return list(distinct(self.faces.intersect(other)))


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
        x, y, z = b - a, c - a, d - a
        yz = Rectangle(a, a + z, a + y + z, a + y)
        xz = Rectangle(a, a + x, a + x + z, a + z)
        xy = Rectangle(a, a + x, a + x + y, a + y)
        super(Cuboid, self).__init__(yz, xz, xy, yz + x, xz + y, xy + z, **kwargs)


class PolygonCollection(PointCollection):
    """A collection of polygons with the same number of vertices.

    Parameters
    ----------
    *args
        The collections of points that define the vertices of the polygons or a (nested) sequence of vertex coordinates.
    **kwargs
        Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    """

    def __init__(self, *args, **kwargs):
        if len(args) > 1:
            args = tuple(a.array for a in args)
            args = np.broadcast_arrays(*args)
            kwargs["copy"] = False
            super(PolygonCollection, self).__init__(np.stack(args, axis=-2), **kwargs)
        else:
            super(PolygonCollection, self).__init__(args[0], **kwargs)
        self._plane = join(*self.vertices[: self.dim]) if self.dim > 2 else None

    @property
    def edges(self):
        """SegmentCollection: The edges of the polygons in the collection."""
        v1 = self.array
        v2 = np.roll(v1, -1, axis=-2)
        result = np.stack([v1, v2], axis=-2)
        return SegmentCollection(result, copy=False)

    def __getitem__(self, index):
        result = super(PolygonCollection, self).__getitem__(index)

        if not isinstance(result, PointCollection):
            return result

        if result.rank == 2:
            return Polygon(result, copy=False)

        return PolygonCollection(result, copy=False)

    def _normalized_projection(self):
        points = self.array

        if self.dim > 2:
            e = self._plane
            o = Point(*[0] * self.dim)
            ind = ~e.contains(o)
            e[ind] = e[ind].parallel(o)
            m = e.basis_matrix
            points = matvec(np.expand_dims(m, -3), points)

        return self._normalize_array(points)

    @property
    def vertices(self):
        """list of PointCollection: The vertices of the polygons."""
        return [
            PointCollection(self.array[..., i, :], copy=False)
            for i in range(self.shape[-2])
        ]

    @property
    def area(self):
        """array_like: The areas of the polygons."""
        points = self._normalized_projection()
        a = sum(
            det(points[..., [0, i, i + 1], :]) for i in range(1, points.shape[-2] - 1)
        )
        return 1 / 2 * np.abs(a)

    def expand_dims(self, axis):
        result = super(PolygonCollection, self).expand_dims(axis)
        if self.dim > 2:
            result._plane = result._plane.expand_dims(axis)
        return result

    def contains(self, other):
        """Tests whether a point or a collection of points is contained in the polygons.

        Parameters
        ----------
        other : Point or PointCollection
            The points to test. If more than one point is given, the shape of the collection must be compatible
            with the shape of the polygon collection.

        Returns
        -------
        numpy.ndarray
            Returns a boolean array of which points are contained in the polygons.

        See Also
        --------
        Polygon.contains

        """
        if other.shape[0] == 0:
            return np.empty((0,), dtype=bool)

        if self.dim > 2:
            coplanar = self._plane.contains(other)

            if not np.any(coplanar):
                return coplanar

            isinf = is_multiple(self._plane.array, infty_hyperplane(self.dim).array, axis=-1)

            # remove coordinates with the largest absolute value in the normal vector
            i = np.argmax(np.abs(self._plane.array[..., :-1]), axis=-1)
            i = np.where(isinf, self.dim, i)
            i = np.expand_dims(i, -1)
            s = self.array.shape
            arr = np.delete(self.array, np.ravel_multi_index(tuple(np.indices(s[:-1])) + (i,), s)).reshape(
                s[:-1] + (-1,))

            if isinstance(other, Point) and i.ndim == 1:
                other = Point(np.delete(other.array, i), copy=False)
            else:
                s = other.shape[:-1] + (1, other.shape[-1])
                other = np.delete(other.array, np.ravel_multi_index(tuple(np.indices(s[:-1])) + (i,), s))
                other = PointCollection(other.reshape(s[:-2] + (-1,)), copy=False)

            # TODO: only test coplanar points
            return coplanar & PolygonCollection(arr, copy=False).contains(other)

        edges = self.edges
        edge_points = edges.contains(PointCollection(np.expand_dims(other.array, -2), copy=False))

        if isinstance(other, PointCollection):
            direction = np.zeros_like(other.array)
            ind = other.isinf
            direction[ind, 0] = other.array[ind, 1]
            direction[ind, 1] = -other.array[ind, 0]
            direction[~ind, 0] = 1
            rays = SegmentCollection(np.stack([other.array, direction], axis=-2), copy=False).expand_dims(-3)
        else:
            direction = [other.array[1], -other.array[0], 0] if other.isinf else [1, 0, 0]
            rays = Segment(np.stack([other.array, direction], axis=-2), copy=False)

        intersections = meet(edges._line, rays._line, _check_dependence=False)

        # ignore edges along the rays
        ray_edges = is_multiple(edges._line.array, rays._line.array, atol=EQ_TOL_ABS, rtol=EQ_TOL_REL, axis=-1)

        # ignore intersections of downward edges that end on the ray
        v1 = edges.array[..., 0, :]
        v2 = edges.array[..., 1, :]
        v1_intersections = (v1[..., 1] <= v2[..., 1]) & is_multiple(intersections.array, v1, atol=EQ_TOL_ABS,
                                                                    rtol=EQ_TOL_REL, axis=-1)
        v2_intersections = (v2[..., 1] <= v1[..., 1]) & is_multiple(intersections.array, v2, atol=EQ_TOL_ABS,
                                                                    rtol=EQ_TOL_REL, axis=-1)

        result = edges.contains(intersections)
        result &= rays.contains(intersections)
        result &= ~ray_edges & ~v1_intersections & ~v2_intersections
        result = np.sum(result, axis=-1) % 2 == 1
        result |= np.any(edge_points, axis=-1)

        return result

    def intersect(self, other):
        """Intersect the polygons with a line, line segment or a collection of lines.

        Parameters
        ----------
        other : Line, Segment, LineCollection or SegmentCollection
            The object to intersect the polygon with.

        Returns
        -------
        PointCollection
            The points of intersection.

        See Also
        --------
        Polygon.intersect

        """
        if isinstance(other, (Segment, SegmentCollection)):
            try:
                result = self._plane.meet(other._line)
            except LinearDependenceError as e:
                if isinstance(other, SegmentCollection):
                    other = other[~e.dependent_values]
                result = self._plane[~e.dependent_values].meet(other._line)
                return result[self[~e.dependent_values].contains(result) & other.contains(result)]
            else:
                return result[self.contains(result) & other.contains(result)]

        try:
            result = self._plane.meet(other)
        except LinearDependenceError as e:
            if isinstance(other, LineCollection):
                other = other[~e.dependent_values]
            result = self._plane[~e.dependent_values].meet(other)
            return result[self[~e.dependent_values].contains(result)]
        else:
            return result[self.contains(result)]


class SegmentCollection(PointCollection):
    """A collection of line segments.

    Parameters
    ----------
    *args
        Two collections of points representing start and endpoints of the line segments or a (nested) sequence of
        coordinates for the start and endpoints.
    **kwargs
        Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    """

    def __init__(self, *args, **kwargs):
        if len(args) == 2:
            a, b = args
            a, b = np.broadcast_arrays(a.array, b.array)
            kwargs["copy"] = False
            super(SegmentCollection, self).__init__(np.stack([a, b], axis=-2), **kwargs)
        else:
            super(SegmentCollection, self).__init__(args[0] if len(args) == 1 else args, **kwargs)

        self._line = join(*self.vertices)

    def __getitem__(self, index):
        result = super(SegmentCollection, self).__getitem__(index)

        if not isinstance(result, PointCollection):
            return result

        if result.rank == 2:
            return Segment(result, copy=False)

        return SegmentCollection(result, copy=False)

    @property
    def vertices(self):
        """list of PointCollection: The start and endpoints of the line segments."""
        a = PointCollection(self.array[..., 0, :], copy=False)
        b = PointCollection(self.array[..., 1, :], copy=False)
        return [a, b]

    @property
    def midpoint(self):
        """PointCollection: The midpoints of the segments."""
        l = self._line.meet(infty_hyperplane(self.dim))
        return harmonic_set(*self.vertices, l)

    @property
    def length(self):
        """array_like: The lengths of the segments."""
        return dist(*self.vertices)

    def expand_dims(self, axis):
        result = super(SegmentCollection, self).expand_dims(axis)
        result._line = result._line.expand_dims(axis - self.dim + 3 if axis < -1 else axis)
        return result

    def contains(self, other, tol=EQ_TOL_ABS):
        """Tests whether a point or a collection of points is contained in the line segments.

        Parameters
        ----------
        other : Point or PointCollection
            The points to test. If more than one point is given, the shape of the collection must be compatible
            with the shape of the segment collection.
        tol : float, optional
            The accepted tolerance.

        Returns
        -------
        numpy.ndarray
            Returns a boolean array of which points are contained in the line segments.

        See Also
        --------
        Segment.contains

        """
        if other.shape[0] == 0:
            return np.empty((0,), dtype=bool)

        result = self._line.contains(other)

        m = self.normalized_array
        arr = matmul(m, m, transpose_b=True)

        b = arr[..., 0]
        c = arr[..., 1]

        # TODO: only project points that lie on the lines
        d = other._matrix_transform(m).array

        # check that crossratio(c-b, b, c, d) is between 0 and 1
        b, c, d = np.broadcast_arrays(b, c, d)
        cd = det(np.stack([c, d], axis=-2))
        bd = det(np.stack([b, d], axis=-2))
        z = -bd
        w = cd - bd
        z_r, z_i = np.real(z), np.imag(z)
        w_r, w_i = np.real(w), np.imag(w)
        x = z_r * w_r + z_i * w_i
        y = w_r ** 2 + w_i ** 2
        x_zero = np.isclose(x, 0, atol=EQ_TOL_ABS)
        y_zero = np.isclose(y, 0, atol=EQ_TOL_ABS)
        return result & (~x_zero | ~y_zero) & (0 <= x + tol) & (x <= y + tol)

    def intersect(self, other):
        """Intersect the line segments with a line, line segment or a collection of lines.

        Parameters
        ----------
        other : Line, Segment, LineCollection or SegmentCollection
            The object to intersect the polygon with.

        Returns
        -------
        PointCollection
            The points of intersection.

        See Also
        --------
        Segment.intersect

        """
        if isinstance(other, (Segment, SegmentCollection)):
            result = meet(self._line, other._line, _check_dependence=False)
            return result[~result.is_zero() & self.contains(result) & other.contains(result)]

        result = meet(self._line, other, _check_dependence=False)
        return result[~result.is_zero() & self.contains(result)]
