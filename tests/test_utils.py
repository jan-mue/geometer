import numpy as np

from geometer.utils import adjugate, det, inv, is_multiple, null_space, orth, roots


def test_is_multiple() -> None:
    a = [1, 2, 3, 4]
    b = [2, 4, 6, 8]
    assert is_multiple(a, b)

    a = [100, 1, 1]
    b = [101, 1, 1]
    assert not is_multiple(a, b)

    a = [0, 0, 0]
    b = np.ones(3)  # type: ignore[assignment]
    assert is_multiple(a, b)
    assert is_multiple(b, a)

    a = [np.eye(3), np.ones((3, 3))]  # type: ignore[list-item]
    b = [2 * np.eye(3), 2 * np.ones((3, 3))]  # type: ignore[list-item]
    assert np.all(is_multiple(a, b, axis=(1, 2)))

    a = [np.ones((3, 3)), 2 * np.ones((3, 3))]  # type: ignore[list-item]
    b = [3 * np.ones((3, 1)), 6 * np.ones((3, 1))]  # type: ignore[list-item]
    assert np.all(is_multiple(a, b, axis=(1, 2)))

    a = [[-0.62996052, 0.62996052, 0.31498026], [0, 0, 2.5198421]]  # type: ignore[list-item]
    b = [[-2, 2, 1], [0, 0, 1]]  # type: ignore[list-item]
    assert np.all(is_multiple(a, b, axis=[1]))  # type: ignore[arg-type]


def test_adjugate() -> None:
    a = np.array(
        [
            [2, 3, 4, 3],
            [5, 1, 7, 6],
            [4, 7, 8, 1],
            [2, 3, 4, 5],
        ]
    )

    adj = adjugate(a)
    assert np.allclose(np.linalg.inv(a) * np.linalg.det(a), adj)


def test_det(rng: np.random.Generator) -> None:
    matrices = rng.random((10, 2, 2))
    assert np.allclose(np.linalg.det(matrices), det(matrices))

    matrices = rng.random((64, 3, 3))
    assert np.allclose(np.linalg.det(matrices), det(matrices))


def test_inv(rng: np.random.Generator) -> None:
    matrices = rng.random((64, 2, 2))
    assert np.allclose(np.linalg.inv(matrices), inv(matrices))

    matrices = rng.random((64, 3, 3))
    assert np.allclose(np.linalg.inv(matrices), inv(matrices))

    matrices = rng.random((64, 4, 4))
    assert np.allclose(np.linalg.inv(matrices), inv(matrices))


def test_orth() -> None:
    a = np.eye(3)
    assert np.allclose(orth(a), a)


def test_null_space() -> None:
    a = np.array(
        [
            [2, 3, 4, 3],
            [5, 1, 7, 6],
            [4, 7, 8, 1],
            [2, 3, 4, 3],
        ]
    )
    assert np.all(abs(a.dot(null_space(a))) < 1e-9)


def test_roots(rng: np.random.Generator) -> None:
    polys = rng.integers(-10, 10, size=(500, 4))
    for p in polys:
        sol = roots(p)
        for x in sol:
            assert np.isclose(np.polyval(p, x), 0)

    assert roots([1, 0, -4]).tolist() == [2, -2]
    assert roots([1, -2]).tolist() == [2]
