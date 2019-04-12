import numpy as np


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
