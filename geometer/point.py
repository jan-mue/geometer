from collections import Iterable

import numpy as np
import sympy
import scipy
from .base import ProjectiveElement, GeometryObject, TensorDiagram, LeviCivitaTensor, Tensor


def join(*args):
    if all(isinstance(o, Point) for o in args):
        n = len(args[0])
        e = LeviCivitaTensor(n, False)
        diagram = TensorDiagram(*[(p, e) for p in args])
        result = diagram.calculate()

        if len(args) == 2:
            return Line(result)
        if len(args) == 3:
            return Plane(result)
        raise ValueError("Wrong number of arguments.")

    if len(args) == 2:
        if isinstance(args[0], Line) and isinstance(args[1], Point):
            return Plane(args[0]*args[1])
        if isinstance(args[0], Point) and isinstance(args[1], Line):
            return Plane(args[1]*args[0])

        l, m = args
        a = (m * l.covariant_tensor).array
        i = np.unravel_index(a.argmax(), a.shape)[0]
        return Plane(a[i, :])


def meet(*args):

    if all(isinstance(o, Plane) for o in args) or all(isinstance(o, Line) for o in args):
        if args[0].tensor_shape[1] == 2:
            l, m = args
            a = (m * l.covariant_tensor).array
            i = np.unravel_index(a.argmax(), a.shape)[1]
            return Point(a[:, i])

        n = len(args[0])
        e = LeviCivitaTensor(n)
        diagram = TensorDiagram(*[(e, x) for x in args])
        result = diagram.calculate()

        if result.tensor_shape[1] == 1:
            return Line(result)
        if result.tensor_shape[0] == 2:
            return Line(result).contravariant_tensor
        if len(args) == n-1:
            return Point(result)
        raise ValueError("Wrong number of arguments.")

    if len(args) == 2:
        if isinstance(args[0], Line) and isinstance(args[1], Plane):
            l, p = args
        elif isinstance(args[1], Line) and isinstance(args[0], Plane):
            p, l = args
        else:
            raise ValueError("Operation not supported.")

        e = LeviCivitaTensor(4)
        diagram = TensorDiagram((e, l), (e, l), (e, p))
        return Point(diagram.calculate())


class Point(ProjectiveElement, GeometryObject):

    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], (Iterable, Tensor)):
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

    def join(self, *others):
        return join(self, *others)

    def intersect(self, other):
        if isinstance(other, Line):
            if other.contains(self):
                return [self]


I = Point([-1j, 1, 0])
J = Point([1j, 1, 0])


class Line(ProjectiveElement, GeometryObject):

    def __init__(self, *args):
        if len(args) == 2:
            pt1, pt2 = args
            super(Line, self).__init__(pt1.join(pt2))
        else:
            super(Line, self).__init__(*args, contravariant_indices=[0])

    def polynomials(self, symbols=None):
        symbols = symbols or sympy.symbols(["x" + str(i) for i in range(self.array.shape[0])])

        def p(row):
            f = sum(x*s for x, s in zip(row, symbols))
            return sympy.Poly(f, symbols)

        if self.dim == 2:
            return [p(self.array)]
        return np.apply_along_axis(p, axis=1, arr=self.array[np.any(self.array, axis=1)])

    @property
    def covariant_tensor(self):
        if self.tensor_shape[0] > 0:
            return self
        e = LeviCivitaTensor(4)
        diagram = TensorDiagram((e, self), (e, self))
        return Line(diagram.calculate())

    @property
    def contravariant_tensor(self):
        if self.tensor_shape[1] > 0:
            return self
        e = LeviCivitaTensor(4, False)
        diagram = TensorDiagram((self, e), (self, e))
        return Line(diagram.calculate())

    def contains(self, pt: Point):
        return self*pt == 0

    def meet(self, other):
        return meet(self, other)

    def join(self, other):
        return join(self, other)

    def is_coplanar(self, other):
        l = other.covariant_tensor
        d = TensorDiagram((l, self), (l, self))
        return d.calculate() == 0

    def intersect(self, other):
        if isinstance(other, Line):
            if self.dim == 3 and not self.is_coplanar(other):
                return []
            return [self.meet(other)]

    def __add__(self, point):
        t = np.array([[1, 0, 0], [0, 1, 0], (-point.normalized()).array]).T
        return Line(self.array.dot(t))

    def __radd__(self, other):
        return self + other

    def __repr__(self):
        return f"Line({str(self.array.tolist())})"

    def parallel(self, through: Point):
        if self.dim == 2:
            p = self.meet(infty)
        elif self.dim == 3:
            p = self.meet(infty_plane)
        else:
            raise NotImplementedError
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
        if self.dim > 2:
            val, vec = scipy.linalg.eigh(self.array.T.dot(self.array))
            return Point(vec[:, np.argmin(val)])

        if np.isclose(self.array[2], 0):
            return Point(0, 0)

        if not np.isclose(self.array[1], 0):
            return Point([0, -self.array[2], self.array[1]])

        return Point([self.array[2], 0, -self.array[0]])

    @property
    def direction(self):
        if self.dim > 2:
            Z = scipy.linalg.null_space(self.array)
            return Point(Z[:, 0] - Z[:, 1])
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


class Plane(ProjectiveElement, GeometryObject):
    
    def __init__(self, *args):
        if len(args) == 2 or len(args) == 3:
            super(Plane, self).__init__(join(*args))
        else:
            super(Plane, self).__init__(*args, contravariant_indices=[0])

    def contains(self, other):
        if isinstance(other, Point):
            return np.isclose(np.vdot(self.array, other.array), 0)
        elif isinstance(other, Line):
            return self*other.covariant_tensor == 0

    def meet(self, *others):
        return meet(self, *others)

    @property
    def basis_matrix(self):
        i = np.where(self.array != 0)[0][0]
        result = np.zeros((4, 3))
        a = [j for j in range(4) if j != i]
        result[i, :] = self.array[a]
        result[a, range(3)] = -self.array[i]
        q, r = np.linalg.qr(result)
        return q.T

    @property
    def polynomial(self):
        symbols = sympy.symbols("x1 x2 x3 x4")
        f = sum(x * s for x, s in zip(self.array, symbols))
        return sympy.Poly(f, symbols)

    def parallel(self, through: Point):
        l = self.meet(infty_plane)
        return join(l, through)

    def intersect(self, other):
        if isinstance(other, Line):
            if self.contains(other):
                return [other]
        return [self.meet(other)]


infty_plane = Plane(0, 0, 0, 1)
