import sympy
from sympy.abc import t, z
import numpy as np
from geometer import Path, residue


def test_cycles():
    p = Path(sympy.exp(2 * sympy.pi * t * 1j), t)
    c = 2*p - p
    assert np.isclose(c.winding_number(0), 1)


def test_length():
    p = Path(sympy.exp(2 * sympy.pi * t * 1j), t)
    assert np.isclose(p.length, 2*np.pi)


def test_cauchy():
    p = Path(sympy.exp(2 * sympy.pi * t * 1j), t)
    assert np.isclose(p.integrate(z**2 - z + 1, z), 0)


def test_winding_number():
    p = Path(sympy.exp(8 * sympy.pi * t * 1j), t)
    assert np.isclose(p.winding_number(0), 4)


def test_residue():
    f = 2/(z-1)
    assert np.isclose(residue(f, z, 1), 2)
