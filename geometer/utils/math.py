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

    If A contains non-trivial numerical objects, the calculation is performed using tensor diagrams. Otherwise this
    method calls numpy.linalg.det.

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


def _exact_null_space(A):
    mat = np.array(A)
    rows, cols = mat.shape

    def row_swap(i, j):
        mat[[i, j]] = mat[[j, i]]

    def cross_cancel(a, i, b, j):
        mat[i] = a * mat[i] - b * mat[j]

    piv_row, piv_col = 0, 0
    pivot_cols = []

    while piv_col < cols and piv_row < rows:
        col = mat[piv_row:, piv_col]

        # find a non-zero element as pivot
        for i, x in np.ndenumerate(col):
            if x != 0:
                pivot_offset, pivot_val = i[0], x
                break
        else:
            piv_col += 1
            continue

        pivot_cols.append(piv_col)
        if pivot_offset != 0:
            row_swap(piv_row, pivot_offset + piv_row)

        # zero above and below the pivot
        for row in range(rows):
            if row == piv_row:
                continue

            val = mat[row, piv_col]

            # if we're already a zero, don't do anything
            if val == 0:
                continue

            cross_cancel(pivot_val, row, val, piv_row)

        piv_row += 1

    # normalize by pivot
    for piv_row, piv_col in enumerate(pivot_cols):
        mat[piv_row, :] /= mat[piv_row, piv_col]

    free_vars = [i for i in range(cols) if i not in pivot_cols]

    basis = np.zeros((cols, len(free_vars)), dtype=mat.dtype)
    for i, free_var in enumerate(free_vars):
        basis[free_var, i] = 1
        for piv_row, piv_col in enumerate(pivot_cols):
            basis[piv_col, i] -= mat[piv_row, free_var]

    return basis


def null_space(A):
    """Constructs an orthonormal basis for the null space of a A using SVD.

    If A contains non-trivial numerical objects, it will fall back to calculating a basis using Gauss elimination.

    Parameters
    ----------
    A : array_like
        The input matrix.

    Returns
    -------
    numpy.ndarray
        Orthonormal basis for the null space of A (as column vectors in the returned matrix).

    """
    try:
        u, s, vh = np.linalg.svd(A, full_matrices=True)
    except TypeError:
        m = _exact_null_space(A)
        q, r = np.linalg.qr(m)
        return q

    cond = np.finfo(s.dtype).eps * max(vh.shape)
    tol = np.amax(s) * cond
    dim = np.sum(s > tol, dtype=int)
    Q = vh[dim:, :].T.conj()
    return Q
