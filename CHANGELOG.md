# Changelog

## [0.4.1](https://github.com/jan-mue/geometer/compare/v0.4.1...v0.4.1) (2025-06-01)

### Bug Fixes

* Rename version file ([#162](https://github.com/jan-mue/geometer/issues/162)) ([5f70f06](https://github.com/jan-mue/geometer/commit/5f70f06a54e8f746ebf69ab80c49263b80f86cb9))
* Use correct artifact name in CI ([#159](https://github.com/jan-mue/geometer/issues/159)) ([8dbadd4](https://github.com/jan-mue/geometer/commit/8dbadd4cdeceb7fb39ce548cf8cfb7247a8e07d6))


### Miscellaneous Chores

* release 0.4.1 ([dd25567](https://github.com/jan-mue/geometer/commit/dd2556741bf00f59acb412b4b0f36a925aa52d11))

## [0.4.0](https://github.com/jan-mue/geometer/compare/v0.3.0...v0.4.0) (2025-06-01)


### Features

* Full type checking with mypy ([#134](https://github.com/jan-mue/geometer/issues/134)) ([8830d29](https://github.com/jan-mue/geometer/commit/8830d2967fd4fff080e6647b881e75b89b2e4c19))
* introduce semantic prs, codespell and add some type hints ([#130](https://github.com/jan-mue/geometer/issues/130)) ([366dcf9](https://github.com/jan-mue/geometer/commit/366dcf9d3f816f47fe0c9d3d57a0c0257aa0398d))
* release package with github action ([bcdad1d](https://github.com/jan-mue/geometer/commit/bcdad1d5b6d0fd5440044872c24f1728811e0c90))
* Upgrade to NumPy 2.x ([#154](https://github.com/jan-mue/geometer/issues/154)) ([9e58b27](https://github.com/jan-mue/geometer/commit/9e58b2725150450ee2e89a50988e05744e4bad2b))
* use release please ([#131](https://github.com/jan-mue/geometer/issues/131)) ([393504e](https://github.com/jan-mue/geometer/commit/393504e60415903196a6a8ac4cf3e458b6c7b4f8))


### Bug Fixes

* Add type annotations for test code ([#142](https://github.com/jan-mue/geometer/issues/142)) ([dca6a7e](https://github.com/jan-mue/geometer/commit/dca6a7eef05d45a494736f1716820eec56cf9705))
* configure release please ([4fd5df1](https://github.com/jan-mue/geometer/commit/4fd5df186bc7877af4e848af25239a2901389454))
* configure release please ([5b257bb](https://github.com/jan-mue/geometer/commit/5b257bbbe099e96152c84bfceac68b205a6f3073))
* Enable additional Ruff rules and fix any issues ([#141](https://github.com/jan-mue/geometer/issues/141)) ([9a45908](https://github.com/jan-mue/geometer/commit/9a4590864c2d875c66b03cb4db2ea5b11b32b181))

## 0.4 - unreleased

### New Features

- Add type hints to code base
- Muted warnings for invalid values in crossratio and other functions
- NumPy 1.22+ required
- Dropped support for Python 3.5, 3.6 and 3.7

## 0.3.4 - released (17.4.2022)

### New Features

- Point.normalized_array and PointCollection.normalized_array will return the original array if normalization is not required

### Bug fixes

- Avoid introducing numerical errors in join/meet by normalizing arrays only by a power of two (#55)
- Simplify code in Segment.contains to avoid integer overflow when normalized_array is an integer array

## 0.3.3 - released (16.4.2022)

### Bug fixes

- Avoid division in Segment.contains for better numerical stability (#55)

## 0.3.2 - released (22.1.2022)

### Bug fixes

- Removed an f-string that caused an error in Python 3.5

## 0.3.1 - released (22.1.2022)

### Bug fixes

- Fixed numerical instability in is_multiple function (#54)

## 0.3 - released (8.1.2022)

### New Features

- Removed sympy as dependency
- Added the TensorCollection class and the following subclasses:
    - ProjectiveCollection
    - SubspaceCollection
    - PointCollection
    - LineCollection
    - PlaneCollection
    - TransformationCollection
    - SegmentCollection
    - PolygonCollection
    - QuadricCollection
- Faster intersection of lines with polygons & polyhedra by using the new collections
- All functions in the operators module support collections in addition to the existing types
- Support for TensorCollection objects in tensor diagrams
- Implemented intersection of quadrics with a collection of lines
- Support for Tensor indexing
- Added Tensor.is_zero()
- Added construction of a perpendicular plane through a line on another plane
- Reformatted source code with black
- Added Transformation.from_points_and_conics to map two conics and pairs of three points
- Added Triangle.circumcenter
- Removed the class AlgebraicCurve and the module utils.polynomial
- Added Point.isreal
- Support numpy 1.20, 1.21 and numpy 1.22
- Run tests on Python 3.9 and Python 3.10

### Bug fixes

- Fixed a bug in the calculation of points of intersection of two conics (issue #36)


## 0.2.3 - released (8.7.2020)

### New Features

- Added Tensor.dtype property
- Added parameters to Tensor class to control ndarray creation (e.g. for setting the dtype)
- Quadrics can now be normalized using their (pseudo-) determinant to reduce numerical errors
- The underlying arrays of tensors are copied less often (controlled by copy parameter)
- Implemented addition/subtraction of points to a quadric
- Transformations can be applied to any object of type Tensor
- More robust algorithm for calculating the intersection of conics
- Added a determinant function that is faster for matrices in dimension 2 and 3
- Updated dependencies (up to NumPy 1.19 and SymPy 1.6 now supported)

### Bug fixes

- Fixed error that was raised when integer arrays are normalized in the join/meet function
- Fixed issues caused by an array not being converted to a Point object in Line.base_point and Line.directions
- Correct handling of special cases in the calculation of the crossing number in Polygon.contains
- Trying to raise an arbitrary tensor to power zero now correctly raises an NotImplementedError
- Equality of polytopes is determined correctly even if the vertices are in a different order


## 0.2.2 - released (15.2.2020)

### New Features

- New adjugate function in utils.math
- New algorithms for Segment.contains, Conic.intersect & Conic.from_crossratio

### Bug fixes

- Fixed an issue with numerical stability when intersecting transformed polytopes (issue #24)
- Conic.components uses a better algorithm that should give correct results in all cases
- Quadric.intersect no longer throws a ValueError when a 3D line has only a single point of intersection
- Line.base_point will now try to always return finite points and Line.direction a point at infinity
- Arrays with small component values are handled correctly by the is_multiple function
- Fixed an issue with Polygon.contains that caused the direction used in the method to be close to zero (issue #25)
- Transformation of line in 3D now works correctly
- The functions null_space and orth now use the same threshold values as Matlab for truncating the singular values


## 0.2.1 - released (3.2.2020)

### New Features

- Added properties shape, rank and T to Tensor class
- Tensor instances can be raised to an arbitrary positive power
- Dynamic calculation of center and radius attributes of RegularPolygon instances
- Added RegularPolygon.inradius property
- Polytope is now a subclass of Tensor
- Added functions for generating transforms that perform scaling and reflections
- Added Polygon.centroid property
- Updated numpy to version 1.18

### Bug fixes

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

### Deprecations

- Deprecated AlgebraicCurve, Subspace.polygons, Plane.polygon, Quadric.polygon and the
  module utils.polynomial in preparation of removal of sympy as dependency


## 0.2 - released (15.9.2019)

### New Features

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

### Bug fixes

- Plane.perpendicular now also works for points that lie on the plane
- Addition/Subtraction of subspaces and points works in more cases
- Adding a point at infinity to another point will give a finite point moved in that direction
- Globally accessible tolerance parameters to avoid inaccurate calculations (issue #22)
- Fixed Transformation.from_points


## 0.1.2 - released (24.2.2019)

### New Features

- Optimized performance of Conic, LeviCivitaTensor and TensorDiagram
- More operations are now compatible with higher-dimensional objects
- New Subspace class that can be used to represent subspaces of any dimension
- New repr and copy methods of Tensor
- scipy is no longer a dependency

### Bug fixes

- Rotation in 3D now returns the correct transformation if the axis is not a normalized vector
- Line.perpendicular now also works for points that lie on the line

## 0.1.1 - released (2.2.2019)
