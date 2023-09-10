from geometer.__version__ import __version__
from geometer.curve import Circle, Cone, Conic, Cylinder, Ellipse, Quadric, Sphere
from geometer.operators import (
    angle,
    angle_bisectors,
    crossratio,
    dist,
    harmonic_set,
    is_cocircular,
    is_collinear,
    is_concurrent,
    is_coplanar,
    is_perpendicular,
)
from geometer.point import I, J, Line, Plane, Point, infty, infty_plane, join, meet
from geometer.shapes import Cuboid, Polygon, Polyhedron, Polytope, Rectangle, RegularPolygon, Segment, Simplex, Triangle
from geometer.transformation import (
    Transformation,
    affine_transform,
    identity,
    reflection,
    rotation,
    scaling,
    translation,
)

PointCollection = Point
LineCollection = Line
PlaneCollection = Plane
QuadricCollection = Quadric
TransformationCollection = Transformation
SegmentCollection = Segment
PolygonCollection = Polygon
