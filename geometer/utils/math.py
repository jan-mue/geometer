import numpy as np
import sympy
from numpy.lib.scimath import sqrt as csqrt
from scipy.integrate import quad


def integrate(fn, variable, domain):
    def u(x):
        return sympy.re(fn.evalf(subs={variable: x}))

    def v(x):
        return sympy.im(fn.evalf(subs={variable: x}))

    a = quad(u, *domain)
    b = quad(v, *domain)
    return a[0] + 1j*b[0]


def xgcd(b, a):
    """Calculates the greatest common divisor of a and b and two additional numbers x and y such that d == x*b + y*a.

    :param b: A number.
    :param a: A number.
    :return: A tuple (d, x, y) such that d = gcd(b, a) and d == x*b + y*a.
    """
    x0, x1, y0, y1 = 1, 0, 0, 1
    while a != 0:
        q, b, a = b // a, a, b % a
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    return b, x0, y0


def cubic_roots(c):
    a, b, c, d = c

    if a == 0 and b == 0:  # Linear equation
        return {-d / c}
    
    elif a == 0:  # Quadratic equation
        D = c**2 - 4 * b * d
        D = csqrt(D)
        x1 = (-c + D) / (2 * b)
        x2 = (-c - D) / (2 * b)
        return {x1, x2}

    f = ((3 * c / a) - ((b ** 2) / (a ** 2))) / 3
    g = (((2 * (b ** 3)) / (a ** 3)) - ((9 * b * c) / (a ** 2)) + (27 * d / a)) / 27
    h = ((g ** 2) / 4 + (f ** 3) / 27)

    if f == 0 and g == 0 and h == 0:  # All 3 roots are real and equal
        x = np.cbrt(d / a)
        return {x}

    elif h <= 0:  # All 3 roots are real

        i = np.sqrt(((g ** 2) / 4) - h)
        j = np.cbrt(i)
        k = np.arccos(-(g / (2 * i)))
        L = j * -1
        M = np.cos(k / 3)
        N = np.sqrt(3) * np.sin(k / 3)
        P = (b / (3 * a)) * -1

        x1 = 2 * j * np.cos(k / 3) - (b / (3 * a))
        x2 = L * (M + N) + P
        x3 = L * (M - N) + P

        return {x1, x2, x3}

    elif h > 0:  # One real root and two complex roots
        R = -(g / 2) + np.sqrt(h)
        S = np.cbrt(R)
        T = -(g / 2) - np.sqrt(h)
        U = np.cbrt(T)

        x1 = (S + U) - (b / (3 * a))
        x2 = -(S + U) / 2 - (b / (3 * a)) + (S - U) * np.sqrt(3) * 0.5j
        x3 = np.conj(x2)

        return {x1, x2, x3}
