import sympy
from scipy.integrate import quad


def integrate(fn, variable, domain):
    def u(x):
        return sympy.re(fn.evalf(subs={variable: x}))

    def v(x):
        return sympy.im(fn.evalf(subs={variable: x}))

    a = quad(u, *domain)
    b = quad(v, *domain)
    return a[0] + 1j*b[0]
