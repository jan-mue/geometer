from .point import Point, Line, Plane, join, meet,  I, J, infty, infty_plane
from .shapes import Segment, Polygon, Triangle, Rectangle
from .curve import AlgebraicCurve, Conic, Circle, Quadric
from .transformation import Transformation, rotation, translation
from .operators import crossratio, is_cocircular, is_perpendicular, is_collinear, is_concurrent, is_coplanar, dist, angle, angle_bisectors
from .__version__ import __version__
