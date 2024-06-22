.. geometer documentation master file, created by
   sphinx-quickstart on Tue Dec 25 01:49:08 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to geometer's documentation!
====================================

Geometer is a geometry library for Python that uses projective geometry and numpy for fast geometric computation.
In projective geometry every point in 2D is represented by a three-dimensional vector and every point in 3D
is represented by a four-dimensional vector. This has the following advantages:

- There are points at infinity that can be treated just like normal points.
- Projective transformations are described by matrices but they can also
  represent translations and general affine transformations.
- Two lines have a unique point of intersection if they lie in the same
  plane. Parallel lines have a point of intersection at infinity.
- Points of intersection, planes or lines through given points can be
  calculated using simple cross products or tensor diagrams.
- Special complex points at infinity and cross ratios can be used to calculate
  angles and to construct perpendicular geometric structures.
- Collections of points and lines can be represented by tensors. Their connecting lines
  and intersections can be calculated using fast matrix multiplications.

Most of the computation in the library is done via tensor diagrams (using numpy.einsum).

The source code of the package can be found on GitHub_ and the documentation on
`Read the Docs`_.

.. _GitHub: https://github.com/jan-mue/geometer
.. _Read the Docs: https://geometer.readthedocs.io

Installation
------------

You can install the package directly from PyPI::

   pip install geometer


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   source/modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

References
----------
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
