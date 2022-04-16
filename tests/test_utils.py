import numpy as np
import pytest

from geometer.utils import adjugate, det, inv, is_multiple, null_space, reduce_multiples, roots


def test_is_multiple():
    a = [1, 2, 3, 4]
    b = [2, 4, 6, 8]
    assert is_multiple(a, b)

    a = [100, 1, 1]
    b = [101, 1, 1]
    assert not is_multiple(a, b)

    a = [0, 0, 0]
    b = np.ones(3)
    assert is_multiple(a, b)
    assert is_multiple(b, a)

    a = [np.eye(3), np.ones((3, 3))]
    b = [2 * np.eye(3), 2 * np.ones((3, 3))]
    assert np.all(is_multiple(a, b, axis=(1, 2)))

    a = [np.ones((3, 3)), 2 * np.ones((3, 3))]
    b = [3 * np.ones((3, 1)), 6 * np.ones((3, 1))]
    assert np.all(is_multiple(a, b, axis=(1, 2)))

    a = [[-0.62996052, 0.62996052, 0.31498026], [0, 0, 2.5198421]]
    b = [[-2, 2, 1], [0, 0, 1]]
    assert np.all(is_multiple(a, b, axis=[1]))


def test_adjugate():
    a = np.array([[2, 3, 4, 3], [5, 1, 7, 6], [4, 7, 8, 1], [2, 3, 4, 5]])

    adj = adjugate(a)
    assert np.allclose(np.linalg.inv(a) * np.linalg.det(a), adj)


def test_det():
    matrices = np.random.rand(10, 2, 2)
    assert np.allclose(np.linalg.det(matrices), det(matrices))

    matrices = np.random.rand(64, 3, 3)
    assert np.allclose(np.linalg.det(matrices), det(matrices))


def test_inv():
    matrices = np.random.rand(64, 2, 2)
    assert np.allclose(np.linalg.inv(matrices), inv(matrices))

    matrices = np.random.rand(64, 3, 3)
    assert np.allclose(np.linalg.inv(matrices), inv(matrices))

    matrices = np.random.rand(64, 4, 4)
    assert np.allclose(np.linalg.inv(matrices), inv(matrices))


def test_null_space():
    a = np.array([[2, 3, 4, 3],
                  [5, 1, 7, 6],
                  [4, 7, 8, 1],
                  [2, 3, 4, 3]])
    assert np.all(abs(a.dot(null_space(a))) < 1e-9)


def test_roots():
    np.random.seed(0)
    polys = np.random.randint(-10, 10, size=(500, 4))
    for p in polys:
        sol = roots(p)
        for x in sol:
            assert np.isclose(np.polyval(p, x), 0)


def test_reduce_multiples():
    a = [3, -6, 12]
    b = [0.7 * 2**3, 0.7 * 2**4, 0.7 * 2**2]
    c = [0.7j + 0.7 * 2**3, 0.7j * 2**3 + 0.7 * 2**4, 0.7j * 2 - 0.7 * 2**2]
    d = [True, False, False]

    assert reduce_multiples(a).tolist() == [1, -2, 4]
    assert reduce_multiples(b).tolist() == [0.7 / 2, 0.7, 0.7 / 4]
    assert reduce_multiples(c).tolist() == [0.7j / 16 + 0.7 / 2, 0.7j / 2 + 0.7, 0.7j / 8 - 0.7 / 4]
    assert reduce_multiples(d).tolist() == d

    with pytest.raises(NotImplementedError, match="Array with dtype str32 cannot be reduced."):
        reduce_multiples(["3", "2", "1"])
