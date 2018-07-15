from numpy.polynomial import polynomial as pl


def polyval(x, c):
    if len(x) != len(c.shape):
        raise ValueError("Dimension of point and polynomial do not match.")
    for xi in x:
        c = pl.polyval(xi, c, tensor=False)
    return c
