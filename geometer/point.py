from collections import Iterable

import numpy as np
import sympy
from .base import ProjectiveElement, TensorDiagram, LeviCivitaTensor


def join(*args):
    n = len(args[0])
    e = LeviCivitaTensor(n, covariant=False)
    diagram = TensorDiagram(*[(p, e) for p in args])
    result = diagram.calculate()

    if sum(a.dim for a in args) == 2:
        return Line(result)
    else:
        return Plane(result)


def meet(*args):
    n = len(args[0])
    e = LeviCivitaTensor(n)
    diagram = TensorDiagram(*[(e, l) for l in args])
    result = diagram.calculate()

    # TODO: fix type of returned object
    if n == 3:
        return Point(result)
    else:
        return Line(result)


class Point(ProjectiveElement):

    dim = 1

    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], Iterable):
            super(Point, self).__init__(*args)
        else:
            super(Point, self).__init__(*args, 1)

    def __add__(self, other):
        result = self.normalized().array[:-1] + other.normalized().array[:-1]
        result = np.append(result, 1)
        return Point(result)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if not np.isscalar(other):
            raise NotImplementedError
        result = self.array[:-1] * other
        result = np.append(result, self.array[-1])
        return Point(result)

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return (-1)*self

    def __repr__(self):
        if self.array[-1] == 0:
            pt = self
        else:
            pt = self.normalized()
        return f"Point({','.join(pt.array[:-1].astype(str))})" + (" at Infinity" if self.array[-1] == 0 else "")

    def normalized(self):
        if self.array[-1] == 0:
            return self.__class__(self.array)
        return self.__class__(self.array / self.array[-1])

    @property
    def x(self):
        return self.normalized().array[0]

    @property
    def y(self):
        return self.normalized().array[1]

    def join(self, other):
        return join(self, other)

    def intersect(self, other):
        if isinstance(other, Line):
            if other.contains(self):
                return [self]


I = Point([-1j, 1, 0])
J = Point([1j, 1, 0])


class Line(ProjectiveElement):

    dim = 2

    def __init__(self, *args):
        if len(args) == 2:
            pt1, pt2 = args
            super(Line, self).__init__(pt1.join(pt2).array, contravariant_indices=[0])
        else:
            super(Line, self).__init__(*args, contravariant_indices=[0])

    @property
    def polynomial(self):
        symbols = sympy.symbols("x y z")
        f = sum(x*s for x,s in zip(self.array, symbols))
        return sympy.Poly(f, symbols)

    def contains(self, pt:Point):
        return np.isclose(np.vdot(self.array, pt.array), 0)

    def meet(self, other):
        return meet(self, other)

    def intersect(self, other):
        if isinstance(other, Line):
            return [self.meet(other)]

    def __add__(self, point):
        t = np.array([[1, 0, 0], [0, 1, 0], (-point.normalized()).array]).T
        return Line(self.array.dot(t))

    def __radd__(self, other):
        return self + other

    def __repr__(self):
        return f"Line({','.join(self.array.astype(str))})"

    def parallel(self, through: Point):
        l = Line(0,0,1)
        p = self.meet(l)
        return p.join(through)

    def is_parallel(self, other):
        p = self.meet(other)
        return np.isclose(p.array[-1], 0)

    def perpendicular(self, through: Point):
        return self.mirror(through).join(through)

    def project(self, pt: Point):
        l = self.mirror(pt).join(pt)
        return self.meet(l)

    @property
    def base_point(self):
        if np.isclose(self.array[2], 0):
            return Point(0, 0)
        elif not np.isclose(self.array[1], 0):
            return Point([0, -self.array[2], self.array[1]])
        else:
            return Point([self.array[2], 0, -self.array[0]])

    @property
    def direction(self):
        if np.isclose(self.array[0], 0) and np.isclose(self.array[1], 0):
            return Point([0, 1, 0])
        return Point(self.array[1], -self.array[0])

    def basic_coeffs(self, pt: Point):
        a = self.base_point.array
        b = np.cross(self.array, a)
        # calculate coordinates of projection of pt on the orthonormal basis {a/|a|, b/|b|}
        return np.vdot(pt.array, a)/np.vdot(a, a), np.vdot(pt.array, b)/np.vdot(b, b)

    def mirror(self, pt: Point):
        l1 = I.join(pt)
        l2 = J.join(pt)
        p1 = self.meet(l1)
        p2 = self.meet(l2)
        m1 = p1.join(J)
        m2 = p2.join(I)
        return m1.meet(m2)


infty = Line(0, 0, 1)


class Plane(ProjectiveElement):

    dim = 3

    def contains(self, pt):
        return np.isclose(np.vdot(self.array, pt.array), 0)

    def intersect(self, other):
        pass
