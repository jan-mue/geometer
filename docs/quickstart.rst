
Quickstart
==========

.. currentmodule:: geometer

Geometry in Two Dimensions
--------------------------

The elementary objects of projective geometry, points and lines can be created using the :obj:`~point.Point` and
:obj:`~point.Line` classes without having to specify the dimension:

.. code-block:: python

    from geometer import *

    p = Point(2, 4)
    q = Point(3, 5)
    l = Line(p, q)
    m = Line(0, 1, 0)

Here we specified a line once by two base points and once using the homogeneous coordinates of the line.
The two elementary operations :obj:`~point.meet` and :obj:`~point.join` can also be called exactly as
one would expect:

.. code-block:: python

    l = join(p, q)
    r = meet(l, m)

Geometer can also construct parallel and perpendicular lines:

.. code-block:: python

    m = l.parallel(through=Point(1, 1))
    n = l.perpendicular(through=Point(1, 1))
    is_perpendicular(m, n)

The function :func:`~operators.is_perpendicular` returns `True` when two lines are perpendicular.
Other angles an distances can also be calculated:

.. code-block:: python

    a = angle(l, Point(1, 0))
    dist(l, p)

    import numpy as np
    p + 2*dist(p, q)*Point(np.cos(a), np.sin(a))

Projective transformations can be easily created using the methods :obj:`~transformation.rotation` and
:obj:`~transformation.translation` or by supplying a matrix to the :class:`~transformation.Transformation` class:

.. code-block:: python

    t1 = translation(0, -1)
    t2 = rotation(-np.pi)
    t3 = Transformation([[0, 1, 0],
                         [1, 0, 0],
                         [0, 0, 1]])
    t1*t2*p

Geometer also includes tools to work with conics. They can be created using the classes :obj:`~curve.Conic` or
:obj:`~curve.Circle`:

.. code-block:: python

    a = Point(-1, 0)
    b = Point(0, 3)
    c = Point(1, 2)
    d = Point(2, 1)
    e = Point(0, -1)

    conic = Conic.from_points(a, b, c, d, e)

To calculate cross ratios of points or lines, the function :func:`~operators.crossratio` can be used:

.. code-block:: python

    t = rotation(np.pi/16)
    crossratio(q, t*q, t**2 * q, t**3 * q, p)

Other interesting operators are :func:`~operators.harmonic_set`, :func:`~operators.angle_bisectors`,
:func:`~operators.is_cocircular` and :func:`~operators.is_collinear`.

Geometry in Three Dimensions
----------------------------

Creating points and lines in 3D works the same as in the two dimensional case:

.. code-block:: python

    p1 = Point(1, 1, 0)
    p2 = Point(2, 1, 0)
    p3 = Point(3, 4, 0)
    l = Line(p1, p2)

In addition to points and lines, in three dimensions we can use the :obj:`~point.Plane` class or the join operation
to create planes:

.. code-block:: python

    A = join(l, p3)
    A.project(Point(3, 4, 5))

Points can be projected onto planes and lines. The result is still a point in 3D but now lying on the plane.

The point of intersection of a plane and a line can be calculated with the :obj:`~point.meet` operation:

.. code-block:: python

    l = Line(Point(1, 2, 3), Point(3, 4, 5))
    A.meet(l)

All other operations such as angles, distances, perpendicular lines, cross ratios are also compatible with objects in
3D. For information on their usage, you can look at the two dimensional example.
