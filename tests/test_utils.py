import numpy as np

from geometer.utils import adjugate, det, inv, null_space, roots


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
