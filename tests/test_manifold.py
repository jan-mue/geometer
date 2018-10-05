import sympy
import numpy as np
from geometer import ComplexCurve, residue


def test_length():
    t = sympy.symbols("t")
    c = ComplexCurve(sympy.exp(2*sympy.pi*t*1j), t)
    assert np.isclose(c.length, 2*np.pi)


def test_cauchy():
    t, z = sympy.symbols("t z")
    c = ComplexCurve(sympy.exp(2 * sympy.pi * t * 1j), t)
    assert np.isclose(c.integrate(z**2 - z + 1, z), 0)


def test_winding_number():
    t = sympy.symbols("t")
    c = ComplexCurve(sympy.exp(8 * sympy.pi * t * 1j), t)
    assert np.isclose(c.winding_number(0), 4)


def test_residue():
    z = sympy.symbols("z")
    f = 2/(z-1)
    assert np.isclose(residue(f, z, 1), 2)
