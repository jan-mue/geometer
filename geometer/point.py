from collections import Iterable

import numpy as np
import sympy


def join(*args):
    if len(args[0]) == 3:
        p, q = args
        return Line(np.cross(p.array, q.array))
    if len(args[0]) == 4:
        if len(args) == 3:
            q = np.array([p.array for p in args]).T
            a, b, c, d = q
            n = (np.linalg.det([b,c,d]), -np.linalg.det([a,c,d]), np.linalg.det([a,b,d]), -np.linalg.det([a,b,c]))
            return Plane(n)


def meet(*args):
    if len(args[0]) == 3:
        l, m = args
        return Point(np.cross(l.array, m.array))
    if len(args[0]) == 4:
        if len(args) == 3:
            q = np.array([p.array for p in args]).T
            a, b, c, d = q
            n = (np.linalg.det([b,c,d]), -np.linalg.det([a,c,d]), np.linalg.det([a,b,d]), -np.linalg.det([a,b,c]))
            return Point(n)


class ProjectiveElement:

    def __init__(self, *args):
        if len(args) == 1:
            self.array = np.atleast_1d(args[0])
        else:
            self.array = np.array([*args])

    def __eq__(self, other):
        pq = self.array.dot(other.array)
        return np.isclose(float(pq**2), float(self.array.dot(self.array)*other.array.dot(other.array)))

    def __len__(self):
        return len(self.array)


class Point(ProjectiveElement):
    
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], Iterable):
            super(Point, self).__init__(*args)
        else:
            super(Point, self).__init__(*args, 1)

    def __add__(self, other):
        result = self.array[:-1] + other.array[:-1]
        result = np.append(result, 1)
        return Point(result)

    def __sub__(self, other):
        result = self.array[:-1] - other.array[:-1]
        result = np.append(result, 1)
        return Point(result)

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

    def join(self, other):
        return join(self, other)


I = Point([-1j, 1, 0])
J = Point([1j, 1, 0])


class Line(ProjectiveElement):

    def __init__(self, *args):
        if len(args) == 2:
            pt1, pt2 = args
            self.array = pt1.join(pt2).array
        else:
            super(Line, self).__init__(*args)

    @property
    def polynomial(self):
        return sum(x * s for x, s in zip(self.array, sympy.symbols("x y z")))

    def contains(self, pt:Point):
        return np.isclose(float(self.array.dot(pt.array)), 0)

    def meet(self, other):
        return meet(self, other)

    def __add__(self, point):
        t = np.array([[1, 0, 0], [0, 1, 0], (-point).array]).T
        return Line(self.array.dot(t))

    def __radd__(self, other):
        return self + other

    def __repr__(self):
        return f"Line({','.join(self.array.astype(str))})"

    def parallel(self, through: Point):
        l = Line(0,0,1)
        p = self.meet(l)
        return p.join(through)

    def perpendicular(self, through: Point):
        return self.mirror(through).join(through)

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

    def contains(self, pt):
        return np.isclose(float(self.array.dot(pt.array)), 0)