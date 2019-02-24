
Changelog
=========

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
