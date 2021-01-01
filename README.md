# geometer

[![image](https://img.shields.io/pypi/v/geometer.svg)](https://pypi.org/project/geometer/)
[![image](https://img.shields.io/pypi/l/geometer.svg)](https://pypi.org/project/geometer/)
[![image](https://img.shields.io/pypi/pyversions/geometer.svg)](https://pypi.org/project/geometer/)
[![Build Status](https://github.com/jan-mue/geometer/workflows/build/badge.svg?branch=master)](https://github.com/jan-mue/geometer/actions)
[![docs](https://readthedocs.org/projects/geometer/badge/?version=latest)](https://geometer.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/github/jan-mue/geometer/coverage.svg?branch=master)](https://codecov.io/github/jan-mue/geometer)
[![Downloads](https://pepy.tech/badge/geometer)](https://pepy.tech/project/geometer)

Geometer is a geometry library for Python 3 that uses projective geometry and numpy for fast geometric computation.
In projective geometry every point in 2D is represented by a three-dimensional vector and every point in 3D
is represented by a four-dimensional vector. This has the following advantages:

- There are points at infinity that can be treated just like normal points.
- Projective transformations are described by matrices but they can also
  represent affine transformations i.e. also translations.
- Every two lines have a unique point of intersection if they lie in the same
  plane. Parallel lines have a point of intersection at infinity.
- Points of intersection, planes or lines through given points can be
  calculated using simple cross products or tensor diagrams.
- Special complex points at infinity and cross ratios can be used to calculate
  angles and to construct perpendicular geometric structures.

Most of the computation in the library is done via tensor diagrams (using numpy.einsum).

Geometer was originally built as a learning exercise and is based on two graduate courses taught at the
Technical University Munich. After investing a lot of time in the project, it is now reasonably well tested
and the API should be stable.

The source code of the package can be found on [GitHub](https://github.com/jan-mue/geometer)
and the documentation on [Read the Docs](https://geometer.readthedocs.io).

## Installation

You can install the package directly from PyPI:
```bash
pip install geometer
```

## Usage

```Python
from geometer import *
import numpy as np

# Meet and Join operations
p = Point(2, 4)
q = Point(3, 5)
l = Line(p, q)
m = Line(0, 1, 0)
l.meet(m)
# Point(-2, 0)

# Parallel and perpendicular lines
m = l.parallel(through=Point(1, 1))
n = l.perpendicular(through=Point(1, 1))
is_perpendicular(m, n)
# True

# Angles and distances (euclidean)
a = angle(l, Point(1, 0))
p + 2*dist(p, q)*Point(np.cos(a), np.sin(a))
# Point(4, 6)

# Transformations
t1 = translation(0, -1)
t2 = rotation(-np.pi)
t1*t2*p
# Point(-2, -5)

# Ellipses/Quadratic forms
a = Point(-1, 0)
b = Point(0, 3)
c = Point(1, 2)
d = Point(2, 1)
e = Point(0, -1)

conic = Conic.from_points(a, b, c, d, e)
ellipse = Conic.from_foci(c, d, bound=b)

# Geometric shapes
o = Point(0, 0)
x, y = Point(1, 0), Point(0, 1)
r = Rectangle(o, x, x+y, y)
r.area
# 1

# 3-dimensional objects
p1 = Point(1, 1, 0)
p2 = Point(2, 1, 0)
p3 = Point(3, 4, 0)
l = p1.join(p2)
A = join(l, p3)
A.project(Point(3, 4, 5))
# Point(3, 4, 0)

l = Line(Point(1, 2, 3), Point(3, 4, 5))
A.meet(l)
# Point(-2, -1, 0)

p3 = Point(1, 2, 0)
p4 = Point(1, 1, 1)
c = Cuboid(p1, p2, p3, p4)
c.area
# 6

# Cross ratios
t = rotation(np.pi/16)
crossratio(q, t*q, t**2 * q, t**3 * q, p)
# 1.4408954235712448

# Higher dimensions
p1 = Point(1, 1, 4, 0)
p2 = Point(2, 1, 5, 0)
p3 = Point(3, 4, 6, 0)
p4 = Point(0, 2, 7, 0)
E = Plane(p1, p2, p3, p4)
l = Line(Point(0, 0, 0, 0), Point(1, 2, 3, 4))
E.meet(l)
# Point(0, 0, 0, 0)

```

## References

Many of the algorithms and formulas implemented in the package are taken from
the following books and papers:

- Jürgen Richter-Gebert, Perspectives on Projective Geometry
- Jürgen Richter-Gebert and Thorsten Orendt, Geometriekalküle
- Olivier Faugeras, Three-Dimensional Computer Vision
- Jim Blinn, Lines in Space: The 4D Cross Product
- Jim Blinn, Lines in Space: The Line Formulation
- Jim Blinn, Lines in Space: The Two Matrices
- Jim Blinn, Lines in Space: Back to the Diagrams
- Jim Blinn, Lines in Space: A Tale of Two Lines
- Jim Blinn, Lines in Space: Our Friend the Hyperbolic Paraboloid
- Jim Blinn, Lines in Space: The Algebra of Tinkertoys
- Jim Blinn, Lines in Space: Line(s) through Four Lines
