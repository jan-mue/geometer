
Changelog
=========

0.2.1 - unreleased
------------------

New Features
------------

- Added properties shape, rank and T to Tensor class
- Tensor instances can be raised to an arbitrary positive power
- Dynamic calculation of center and radius attributes of RegularPolygon instances
- Added RegularPolygon.inradius property
- Polytope is now a subclass of Tensor
- Added functions for generating transforms that perform scaling and reflections
- Added Polygon.centroid property
- Updated numpy to version 1.18

Bug fixes
---------

- Transformations are now applied correctly to quadrics and conics
- Fixed bug that made transformation of Cuboid & RegularPolygon fail (issue #23)
- Raising transformations to a power (other than 1) is calculated correctly
- Tolerance parameters are correctly used in Tensor.__eq__
- Scalar multiplication with Points is calculated correctly using normalized_array
- Fixed copy method Tensor subclasses
- Return real angles instead of angles with complex type
- Fixed init method of regular polygons that aren't centered at the origin
- Indices passed to Tensor constructor are validated and negative indices converted
- Fixed init method of Cone & Cylinder classes


0.2 - released (15.9.2019)
--------------------------

New Features
------------

- New shapes module that implements line segments, polygons and general polytopes
- New Sphere class (a subclass of Quadric) that works in any dimension
- New classes representing a cone and a cylinder in 3D
- Tensor has a new tensor_product method to calculate the tensor product with another tensor
- Ellipse class that constructs a conic from center and radius
- Added Conic.foci and Conic.polar
- Construct a conic from its focal points, using a tangent line or a cross ratio
- Faster and more general intersect method for quadrics
- Refactored & documented the code for calculation of tensor diagrams
- New KroneckerDelta tensor
- TensorDiagram calculates results differently, using free indices from front to back
- New method TensorDiagram.add_node to add tensors without edge to the diagram
- Added Circle.intersection_angle to calculate the angle of intersection of two circles
- is_perpendicular now works with two planes
- New function is_multiple in utils module

Bug fixes
---------

- Plane.perpendicular now also works for points that lie on the plane
- Addition/Subtraction of subspaces and points works in more cases
- Adding a point at infinity to another point will give a finite point moved in that direction
- Globally accessible tolerance parameters to avoid inaccurate calculations (issue #22)
- Fixed Transformation.from_points


0.1.2 - released (24.2.2019)
----------------------------

New Features
------------

- Optimized performance of Conic, LeviCivitaTensor and TensorDiagram
- More operations are now compatible with higher-dimensional objects
- New Subspace class that can be used to represent subspaces of any dimension
- New repr and copy methods of Tensor
- scipy is no longer a dependency

Bug fixes
---------

- Rotation in 3D now returns the correct transformation if the axis is not a normalized vector
- Line.perpendicular now also works for points tha lie on the line

0.1.1 - released (2.2.2019)
---------------------------
