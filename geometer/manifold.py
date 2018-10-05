import sympy
import numpy as np
from .utils import integrate


def residue(f, variable, z):
    t = sympy.symbols("t")
    c = ComplexCurve(z + sympy.exp(2*sympy.pi*t*1j), t)
    return 1/(2*np.pi*1j)*c.integrate(f, variable)


class Manifold:
    pass


class ComplexCurve(Manifold):

    def __init__(self, parametrization, variable, domain=(0, 1)):
        self.parametrization = parametrization
        self.variable = variable
        self.domain = domain

    @property
    def length(self):
        return integrate(abs(sympy.diff(self.parametrization, self.variable)), self.variable, self.domain)

    def integrate(self, f, variable):
        dy = sympy.diff(self.parametrization, self.variable)
        return integrate(f.subs(variable, self.parametrization)*dy, self.variable, self.domain)

    def winding_number(self, pt):
        z = sympy.symbols("z")
        return 1/(2*np.pi*1j)*self.integrate(1/(z-pt), z)

    def __add__(self, other):
        if not isinstance(other, ComplexCurve):
            return NotImplemented
        return Cycle(self, other)

    def __neg__(self):
        return ComplexCurve(self.parametrization.subs(self.variable, self.domain[1]-self.variable), self.variable, self.domain)

    def __sub__(self, other):
        if not isinstance(other, ComplexCurve):
            return NotImplemented
        return self + (-other)


class Cycle:

    def __init__(self, *curves):
        self.curves = list(curves)

    @property
    def length(self):
        return sum(c.length for c in self.curves)

    def integrate(self, f, variable):
        return sum(c.integrate(f, variable) for c in self.curves)
