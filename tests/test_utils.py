from geometer.utils import null_space
import numpy as np


def test_null_space():
    a = np.array([[2, 3, 4, 3],
                  [5, 1, 7, 6],
                  [4, 7, 8, 1],
                  [2, 3, 4, 3]])
    assert np.all(abs(a.dot(null_space(a))) < 1e-9)
