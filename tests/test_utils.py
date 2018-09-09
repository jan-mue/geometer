from geometer.utils import cubic_roots
import numpy as np


def test_cubic_roots():
    polys = np.random.randint(-10, 10, size=(500, 4))
    for p in polys:
        sol = cubic_roots(p)
        for x in sol:
            assert np.isclose(np.polyval(p, x), 0)
