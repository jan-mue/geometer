from itertools import combinations

import numpy as np

from .base import TensorDiagram, Tensor, LeviCivitaTensor
from .utils import distinct, is_multiple
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

    Attributes
    ----------
    array : numpy.ndarray
        The underlying numpy array.

    """

    def __init__(self, *args):
        if len(args) == 1:
            self.array = np.array(args[0])
        else:
            self.array = np.array([a.array for a in args])

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, ", ".join(str(v) for v in self.vertices))

    @property
    def dim(self):
        """int: The dimension of the space that the polytope lives in."""
        return self.array.shape[-1] - 1

    @property
    def vertices(self):
        """list of Point: The vertices of the polytope."""
        vertices = self.array.reshape(-1, self.array.shape[-1])
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

            if self.array.shape != other.array.shape:
                return False

            if self.array.ndim > 2:
                # facets equal up to reordering
                return all(f in other.facets for f in self.facets) and all(f in self.facets for f in other.facets)

            return np.all(is_multiple(self.array, other.array, axis=-1))

        return NotImplemented

    def __add__(self, other):
        if isinstance(other, Point):
            return type(self)(*[f + other for f in self.facets])
        return NotImplemented

    def __radd__(self, other):
        return self + other


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

    @staticmethod
    def _contains(vertex1, vertex2, points):

        # allow multiple points for one vertex
        if points.ndim != vertex1.ndim:
            vertex1 = np.tile(vertex1, (len(points), 1))
            vertex2 = np.tile(vertex2, (len(points), 1))

        v1inf = np.isclose(vertex1.T[-1], 0)
        v2inf = np.isclose(vertex2.T[-1], 0)
        pinf = np.isclose(points.T[-1], 0)

        d1 = np.empty(pinf.shape)
        d2 = np.empty(pinf.shape)

        direction = np.empty(vertex1.shape, np.common_type(vertex1, vertex2))
        result = np.empty(pinf.shape, dtype=bool)

        # when possible, we will use a finite point as start point of a ray/segment
        start = vertex1.copy()

        # normalize finite points
        ind = ~pinf
        points[ind] = (points[ind].T / points[ind, -1]).T

        # only vertex1 at infinity
        ind1 = v1inf & (~v2inf)
        if np.sum(ind1) > 0:
            direction[ind1] = vertex1[ind1]
            start[ind1] = (vertex2[ind1].T / vertex2[ind1, -1]).T
            d2[ind1] = np.inf
            ind1 &= pinf
            result[ind1] = is_multiple(vertex1[ind1], points[ind1], axis=-1)

        # only vertex2 at infinity
        ind2 = v2inf & (~v1inf)
        if np.sum(ind2) > 0:
            direction[ind2] = vertex2[ind2]
            start[ind2] = (vertex1[ind2].T / vertex1[ind2, -1]).T
            d2[ind2] = np.inf
            ind2 &= pinf
            result[ind2] = is_multiple(vertex2[ind2], points[ind2], axis=-1)

        # both vertices finite
        ind3 = ~v1inf & ~v2inf
        if np.sum(ind3) > 0:
            v1 = vertex1[ind3].T / vertex1[ind3, -1]
            v2 = vertex2[ind3].T / vertex2[ind3, -1]
            direction[ind3] = (v2 - v1).T
            start[ind3] = v1.T
            d2[ind3] = np.sum(direction[ind3] ** 2, axis=-1)
            ind3 &= pinf
            result[ind3] = False

        # both vertices at infinity
        ind4 = v1inf & v2inf
        if np.sum(ind4) > 0:
            direction[ind4] = vertex2[ind4] - vertex1[ind4]
            d2[ind4] = np.sum(direction[ind4] ** 2, axis=-1)
            ind4 &= ~pinf
            result[ind4] = False

        # calculate result for remaining cases
        ind = ~(ind1 | ind2 | ind3 | ind4)
        d1[ind] = np.sum((points[ind] - start[ind]) * direction[ind], axis=-1)
        result[ind] = (0 <= d1[ind]) & (d1[ind] <= d2[ind])

        return result

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

        return self._contains(*self.array, other.array)

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

    def __new__(cls, *args, **kwargs):
        if len(args) == 2 and all(isinstance(x, Point) for x in args):
            return Segment(*args)

        return super(Polytope, cls).__new__(cls)

    def __init__(self, *args):
        if all(isinstance(x, Point) for x in args):
            args = [Simplex(*x) for x in combinations(args, len(args)-1)]

        super(Simplex, self).__init__(*args)

    @property
    def volume(self):
        """float: The volume of the simplex, calculated using the Cayley–Menger determinant."""
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
        if all(isinstance(x, Segment) for x in args):
            args = [s.vertices[0] for s in args]
        super(Polygon, self).__init__(*args)
        self._plane = None
        if self.dim > 2:
            self._plane = join(*self.vertices[:self.dim])
            for s in self.vertices[self.dim:]:
                if not self._plane.contains(s):
                    raise NotCoplanar("The vertices of a polygon must be coplanar!")

    @property
    def facets(self):
        segments = []
        for i in range(len(self.array)):
            segments.append(Segment(self.array[[i, (i + 1) % len(self.array)]]))
        return segments

    @property
    def edges(self):
        return self.facets

    def _intersect_coplanar_line(self, line):
        n = self.array.shape[-1]

        e1 = LeviCivitaTensor(n, False)
        v1 = self.array
        v2 = np.roll(self.array, -1, axis=0)

        vertices = Tensor(np.einsum('ij,ik->ijk', v1, v2), covariant=[1, 2])
        diagram = TensorDiagram((vertices, e1), (vertices, e1))

        e2 = LeviCivitaTensor(n)

        for i in range(n-2):
            diagram.add_edge(e2, e1)

        diagram.add_edge(e2, line)
        points = diagram.calculate().array

        if n > 3:
            i = np.unravel_index(np.abs(points).argmax(), points.shape)
            points = points[(slice(None), slice(None)) + i[2:]]

        points = points.T
        ind = Segment._contains(v1, v2, points)
        return points[ind]

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
                direction = Point([0, 1] + [0] * (self.dim-1))
        elif self.dim == 2:
            direction = Point([1, 0, 0])
        else:
            a = self._plane.array
            if np.isclose(a[0], 0):
                direction = Point([1] + [0]*self.dim)
            else:
                direction = Point([a[1], -a[0]] + [0]*(self.dim-1))

        ray = Segment(other, direction)
        intersections = self._intersect_coplanar_line(ray._line)
        intersection_count = np.sum(Segment._contains(*ray.array, intersections))

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

        if isinstance(other, Segment):
            intersections = self._intersect_coplanar_line(other._line)
            intersections = intersections[Segment._contains(*other.array, intersections)]
        else:
            intersections = self._intersect_coplanar_line(other)

        return list(distinct(Point(x) for x in intersections))

    @property
    def area(self):
        """float: The area of the polygon."""
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

    def __init__(self, *args):
        super(Triangle, self).__init__(*args)


class Rectangle(Polygon):
    """A class representing rectangles.

    Parameters
    ----------
    a : Point
    b : Point
    c : Point
    d : Point

    """

    def __init__(self, *args):
        super(Rectangle, self).__init__(*args)


class Polyhedron(Polytope):
    """A class representing polyhedra (3-polytopes).

    """

    @property
    def faces(self):
        return self.facets

    @property
    def edges(self):
        return list(distinct(s for f in self.faces for s in f.edges))

    @property
    def area(self):
        """float: The surface area of the polyhedron."""
        return sum(s.area for s in self.faces)

    def _intersect_line(self, line):
        n = self.array.shape[-1]

        e = LeviCivitaTensor(n, False)
        v1 = self.array[:, 0, :]
        v2 = self.array[:, 1, :]
        v3 = self.array[:, 2, :]

        vertices = Tensor(np.einsum('ij,ik,il->ijkl', v1, v2, v3), covariant=[1, 2, 3])
        diagram = TensorDiagram((vertices, e), (vertices, e), (vertices, e))

        planes = diagram.calculate().array

        e = LeviCivitaTensor(n)
        diagram = TensorDiagram((e, Tensor(planes, covariant=[0])), (e, line), (e, line))
        points = diagram.calculate().array.T

        # filter points actually contained in the faces
        direction = np.zeros(points.shape, planes.dtype)
        isinf = np.isclose(points[:, -1], 0)
        direction[isinf, 0] = 1
        ind = is_multiple(direction[isinf], points[isinf], axis=-1)
        direction[isinf, 1] = ind.astype(int)

        direction[~isinf, 3] = 1
        direction[~isinf & np.isclose(planes[:, 0], 0), 0] = 1
        ind = ~isinf & ~np.isclose(planes[:, 0], 0)
        direction[ind, 0] = planes[ind, 1]
        direction[ind, 1] = -planes[ind, 0]

        ind = np.empty(points.shape[0], dtype=bool)

        # TODO: replace this for loop
        for i, (v, pt, d) in enumerate(zip(self.array, points, direction)):
            ray = Segment([pt, d])
            intersections = Polygon(v)._intersect_coplanar_line(ray._line)
            intersection_count = np.sum(Segment._contains(pt, d, intersections))
            ind[i] = intersection_count % 2 == 1

        return points[ind]

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
        if isinstance(other, Segment):
            intersections = self._intersect_line(other._line)
            intersections = intersections[Segment._contains(*other.array, intersections)]
        else:
            intersections = self._intersect_line(other)

        return list(distinct(Point(x) for x in intersections))


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
