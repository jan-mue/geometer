from geometer.utils import poly_to_np_array, np_array_to_poly, null_space, adjugate, det
import numpy as np
from sympy.abc import x, y, z


def test_poly_to_array():
    p = 5*x**2 + 2*x*y + 3*y**2 + z**3
    a = poly_to_np_array(p, (x, y, z))
    assert np.array_equal(a, [[[0, 0, 0, 1], [0, 0, 0, 0], [3, 0, 0, 0], [0, 0, 0, 0]],
                              [[0, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                              [[5, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                              [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])


def test_array_to_poly():
    a = [[[0, 0, 0, 1], [0, 0, 0, 0], [3, 0, 0, 0], [0, 0, 0, 0]],
         [[0, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
         [[5, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
         [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
    assert np_array_to_poly(a, (x, y, z)) == 5*x**2 + 2*x*y + 3*y**2 + z**3


def test_adjugate():
    a = np.array([[2, 3, 4, 3],
                  [5, 1, 7, 6],
                  [4, 7, 8, 1],
                  [2, 3, 4, 5]])

    adj = adjugate(a)
    assert np.allclose(np.linalg.inv(a)*np.linalg.det(a), adj)


def test_det():
    matrices = np.random.rand(10, 2, 2)
    assert np.allclose(np.linalg.det(matrices), det(matrices))

    matrices = np.random.rand(64, 3, 3)
    assert np.allclose(np.linalg.det(matrices), det(matrices))


def test_null_space():
    a = np.array([[2, 3, 4, 3],
                  [5, 1, 7, 6],
                  [4, 7, 8, 1],
                  [2, 3, 4, 3]])
    assert np.all(abs(a.dot(null_space(a))) < 1e-9)
