import sympy
import numpy as np
from .utils import integrate
from .base import GeometryObject
from .point import Point, Line


def residue(f, variable, z):
    t = sympy.symbols("t")
    c = Path(z + sympy.exp(2 * sympy.pi * t * 1j), t)
    return 1/(2*np.pi*1j)*c.integrate(f, variable)


class Manifold(GeometryObject):

    def __init__(self, parametrization, variables, domain):
        self.parametrization = parametrization
        self._variables = variables
        self.domain = domain

    def eval(self, *args):
        f = sympy.lambdify(self._variables, self.parametrization)
        return Point(f(*args))

    def intersect(self, other):
        pass

    def tangent(self, *args):
        pass


class Path(Manifold):

    def __init__(self, parametrization, variable, domain=(0, 1)):
        super(Path, self).__init__(parametrization, variable, domain)

    @property
    def length(self):
        return integrate(abs(sympy.diff(self.parametrization, self._variables)), self._variables, self.domain)

    def integrate(self, f, variable):
        dy = sympy.diff(self.parametrization, self._variables)
        return integrate(f.subs(variable, self.parametrization) * dy, self._variables, self.domain)

    def winding_number(self, pt):
        z = sympy.symbols("z")
        return 1/(2*np.pi*1j)*self.integrate(1/(z-pt), z)

    @property
    def start(self):
        return self.eval(self.domain[0])

    @property
    def end(self):
        return self.eval(self.domain[1])

    def is_closed(self):
        return self.start == self.end

    def __add__(self, other):
        if not isinstance(other, Path):
            return NotImplemented
        return Cycle(self, other)

    def __neg__(self):
        return Path(self.parametrization.subs(self._variables, self.domain[1] - self._variables), self._variables, self.domain)

    def __sub__(self, other):
        if not isinstance(other, Path):
            return NotImplemented
        return self + (-other)

    def __mul__(self, other):
        if not isinstance(other, int) or not self.is_closed():
            return NotImplemented
        return Cycle(*(abs(other)*[-self if other < 0 else self]))

    def __rmul__(self, other):
        return self * other


class Cycle:

    def __init__(self, *paths):
        if len(paths) < 1:
            raise ValueError("A cycle requires at least one path.")
        self._paths = list(paths)

    @property
    def length(self):
        return sum(c.length for c in self._paths)

    def integrate(self, f, variable):
        return sum(c.integrate(f, variable) for c in self._paths)

    def winding_number(self, pt):
        z = sympy.symbols("z")
        return 1/(2*np.pi*1j)*self.integrate(1/(z-pt), z)

    def __add__(self, other):
        if isinstance(other, Path):
            return Cycle(*self._paths, other)
        if isinstance(other, Cycle):
            return Cycle(*self._paths, *other._paths)
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, Path):
            return Cycle(other, *self._paths)
        return NotImplemented

    def __neg__(self):
        return Cycle(*reversed([-c for c in self._paths]))

    def __sub__(self, other):
        if not isinstance(other, Path):
            return NotImplemented
        return self + (-other)

    def __mul__(self, other):
        if not isinstance(other, int):
            return NotImplemented
        c = Cycle(*(abs(other)*self._paths))
        return -c if other < 0 else c

    def __rmul__(self, other):
        return self * other
