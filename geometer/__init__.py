# flake8: noqa

from .__version__ import __version__
from .curve import Circle, Cone, Conic, Cylinder, Ellipse, Quadric, Sphere
from .operators import (angle, angle_bisectors, crossratio, dist, harmonic_set, is_cocircular, is_collinear,
                        is_concurrent, is_coplanar, is_perpendicular)
from .point import I, J, Line, Plane, Point, infty, infty_plane, join, meet
from .shapes import Cuboid, Polygon, Polyhedron, Polytope, Rectangle, RegularPolygon, Segment, Simplex, Triangle
from .transformation import Transformation, affine_transform, identity, reflection, rotation, scaling, translation

PointCollection = Point
LineCollection = Line
PlaneCollection = Plane
QuadricCollection = Quadric
TransformationCollection = Transformation
SegmentCollection = Segment
PolygonCollection = Polygon
