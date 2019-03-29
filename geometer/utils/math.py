import numpy as np


def isclose(a, b):
    try:
        return np.isclose(a, b)
    except TypeError:
        return a == b


def allclose(a, b):
    try:
        return np.allclose(a, b)
    except TypeError:
        return np.all(a == b)


def det(A):
    """Computes the determinant of A.

    Parameters
    ----------
    A : array_like
        The input matrix.

    Returns
    -------
    float
        The determinant of A.

    """
    try:
        return np.linalg.det(A)
    except TypeError:
        from ..base import TensorDiagram, Tensor, LeviCivitaTensor
        e = LeviCivitaTensor(A[0].shape[0], False)
        tensors = [Tensor(a) for a in A]
        return TensorDiagram(*[(t, e) for t in tensors]).calculate().array[0]


def null_space(A):
    """Constructs an orthonormal basis for the null space of a A using SVD.

    Parameters
    ----------
    A : array_like
        The input matrix.

    Returns
    -------
    numpy.ndarray
        Orthonormal basis for the null space of A (as column vectors in the returned matrix).

    """
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    cond = np.finfo(s.dtype).eps * max(vh.shape)
    tol = np.amax(s) * cond
    dim = np.sum(s > tol, dtype=int)
    Q = vh[dim:, :].T.conj()
    return Q
