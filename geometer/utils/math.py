from numpy.polynomial import polynomial as pl


def polyval(x, c):
    if len(x) != len(c.shape):
        raise ValueError("Dimension of point and polynomial do not match.")
    for xi in x:
        c = pl.polyval(xi, c, tensor=False)
    return c


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
    return  b, x0, y0
