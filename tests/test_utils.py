from geometer.utils import null_space, adjugate, det
import numpy as np


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

    matrices = np.random.rand(10, 3, 3)
    assert np.allclose(np.linalg.det(matrices), det(matrices))


def test_null_space():
    a = np.array([[2, 3, 4, 3],
                  [5, 1, 7, 6],
                  [4, 7, 8, 1],
                  [2, 3, 4, 3]])
    assert np.all(abs(a.dot(null_space(a))) < 1e-9)
