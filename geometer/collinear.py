import numpy as np

def is_collinear(*args):
    return np.isclose(np.linalg.det([p.array for p in args]), 0)

is_concurrent = is_collinear
