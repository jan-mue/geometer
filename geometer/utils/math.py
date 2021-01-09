import math
import numpy as np
from numpy.lib.scimath import sqrt as csqrt


def is_multiple(a, b, axis=None, rtol=1.0e-15, atol=1.0e-8):
    """Returns a boolean array where two arrays are scalar multiples of each other along a given axis.

    This function compares the absolute value of the scalar product and the product of the norm of the arrays (along
    an axis). The Cauchy-Schwarz inequality guarantees in its edge case that this equality holds if and only if one
    of the vectors is a scalar multiple of the other.

    For documentation of the tolerance parameters see :func:`numpy.isclose`.

    Parameters
    ----------
    a, b : array_like
        Input arrays to compare.
    axis : None or int or tuple of ints, optional
        The axis or axes along which the two arrays are compared.
        The default axis=None will compare the whole arrays and return only a single boolean value.
    rtol : float, optional
        The relative tolerance parameter.
    atol : float, optional
        The absolute tolerance parameter.

    Returns
    -------
    array_like
        Returns a boolean array of where along the given axis the arrays are a scalar multiple of each other (within the
        given tolerance). If no axis is given, returns a single boolean value.

    """
    a = np.asarray(a)
    b = np.asarray(b)

    a = a / np.max(np.abs(a), axis=axis, keepdims=True)
    b = b / np.max(np.abs(b), axis=axis, keepdims=True)

    if axis is None:
        a = a.ravel()
        b = b.ravel()

    ab = np.sum(a * b.conj(), axis=axis)
    return np.isclose(
        ab * ab.conj(),
        np.sum(a * a.conj(), axis=axis) * np.sum(b * b.conj(), axis=axis),
        rtol,
        atol,
    )


def hat_matrix(*args):
    r"""Builds a skew symmetric matrix with the given scalars in the positions shown below.

    .. math::

        \begin{pmatrix}
            0  &  c & -b\\
            -c &  0 & a \\
            b  & -a & 0
        \end{pmatrix}

    Parameters
    ----------
    a, b, c : float
        The scalars to use in the matrix.

    Returns
    -------
    numpy.ndarray
        The resulting antisymmetric matrix.

    """
    if len(args) == 1:
        args = args[0]

    x = np.asarray(args)
    n = int(1 + np.sqrt(1 + 8 * x.shape[-1])) // 2

    result = np.zeros(x.shape[:-1] + (n, n), x.dtype)

    if n == 3:
        i, j = [1, 2, 0], [2, 0, 1]
        result[..., i, j] = x
        result[..., j, i] = -x
        return result

    i, j = np.triu_indices(n, 1)
    i, j = i[::-1], j[::-1]
    result[..., i, j] = x
    result[..., j, i] = -x

    return result


def _assert_square_matrix(A):
    if A.ndim < 2:
        raise np.linalg.LinAlgError(
            "%s-dimensional array given. Array must be at least two-dimensional"
            % A.ndim
        )
    m, n = A.shape[-2:]
    if m != n:
        raise np.linalg.LinAlgError("Last 2 dimensions of the array must be square")


def adjugate(A):
    r"""Calculates the adjugate matrix of A.

    The resulting matrix is defined by

    .. math::
        \textrm{adj}(A)_{ij} = (-1)^{i+j} M_{j i},

    where :math:`M_{j i}` is the determinant of the submatrix of :math:`A` obtained by deleting the j-th row and the
    i-th column of :math:`A`.

    For small matrices, this function uses the following formula (Einstein notation):

    .. math::
        \textrm{adj}(A)_{ij} = \frac{1}{(n-1)!} \varepsilon_{i\ i_2 \ldots i_n} \varepsilon_{j\ j_2 \ldots j_n} A_{j_2 i_2} \ldots A_{j_n i_n}

    Source (German):
    https://de.wikipedia.org/wiki/Levi-Civita-Symbol#Zusammenhang_mit_der_Determinante

    Parameters
    ----------
    A : (..., M, M) array_like
        The input matrix.

    Returns
    -------
    (..., M, M) numpy.ndarray
        The adjugate of A.

    """
    A = np.asarray(A)
    _assert_square_matrix(A)
    n = A.shape[-1]

    if n <= 1:
        return A

    if n == 2:
        result = A[..., [[1, 0], [1, 0]], [[1, 1], [0, 0]]]
        result[..., [0, 1], [1, 0]] *= -1
        return result

    if n >= 5 or A.size >= n * n * 64:
        indices = np.indices((n, n))
        indices = [
            np.delete(np.delete(indices, i, axis=1), j, axis=2)
            for i in range(n)
            for j in range(n)
        ]
        indices = np.stack(indices, axis=1)
        minors = A[..., indices[0], indices[1]]
        result = det(minors).reshape(A.shape)
        result = np.swapaxes(result, -1, -2)
        result[..., 1::2, ::2] *= -1
        result[..., ::2, 1::2] *= -1
        return result

    from ..base import TensorDiagram, Tensor, LeviCivitaTensor

    e1 = LeviCivitaTensor(n, False)
    e2 = LeviCivitaTensor(n, False)
    tensors = [Tensor(A, tensor_rank=2, copy=False) for _ in range(n - 1)]
    diagram = TensorDiagram(*[(t, e1) for t in tensors], *[(t, e2) for t in tensors])

    return np.swapaxes(diagram.calculate().array, -1, -2) / math.factorial(n - 1)


def det(A):
    """Computes the determinant of A.

    Parameters
    ----------
    A : (..., M, M) array_like
        The input matrix.

    Returns
    -------
    (...) array_like
        The determinant of A.

    """
    A = np.asarray(A)
    _assert_square_matrix(A)
    n = A.shape[-1]

    if n == 2:
        return A[..., 0, 0] * A[..., 1, 1] - A[..., 1, 0] * A[..., 0, 1]

    if n == 3 and A.size >= 9 * 64:
        return (
            A[..., 0, 0] * A[..., 1, 1] * A[..., 2, 2]
            + A[..., 0, 1] * A[..., 1, 2] * A[..., 2, 0]
            + A[..., 0, 2] * A[..., 1, 0] * A[..., 2, 1]
            - A[..., 2, 0] * A[..., 1, 1] * A[..., 0, 2]
            - A[..., 2, 1] * A[..., 1, 2] * A[..., 0, 0]
            - A[..., 2, 2] * A[..., 1, 0] * A[..., 0, 1]
        )

    return np.linalg.det(A)


def inv(A):
    """Computes the inverse of A.

    Parameters
    ----------
    A : (..., M, M) array_like
        The input matrix.

    Returns
    -------
    (..., M, M) numpy.ndarray
        The inverse of A.

    """
    A = np.asarray(A)
    _assert_square_matrix(A)
    n = A.shape[-1]

    if n <= 4 and A.size >= n * n * 64:
        d = det(A)

        if np.any(d == 0):
            raise np.linalg.LinAlgError("Singular matrix")

        return adjugate(A) / d[..., None, None]

    return np.linalg.inv(A)


def null_space(A, dim=None):
    """Constructs an orthonormal basis for the null space of a A using SVD.

    Parameters
    ----------
    A : (..., M, N) array_like
        The input matrix.
    dim : int or None, optional
        The dimension of the null space if previously known.

    Returns
    -------
    (..., N, K) numpy.ndarray
        Orthonormal basis for the null space of A (as column vectors in the returned matrix).

    """
    u, s, vh = np.linalg.svd(A, full_matrices=True)

    if dim is None:
        tol = max(A.shape[-2:]) * np.spacing(np.max(s, axis=-1, keepdims=True))
        dim = np.sum(s > tol, axis=-1, dtype=int)
        if not np.all(dim == dim.flat[0]):
            raise ValueError(
                "Cannot calculate the null spaces of matrices when the spaces have different dimensions."
            )
        dim = -dim.flat[0]

    Q = np.swapaxes(vh[..., -dim:, :], -1, -2).conj()
    return Q


def orth(A, dim=None):
    """Constructs an orthonormal basis for the range of A using SVD.

    Parameters
    ----------
    A : (..., M, N) array_like
        The input matrix.
    dim : int or None, optional
        The dimension of the image space if previously known.

    Returns
    -------
    (..., M, K) numpy.ndarray
        Orthonormal basis for the range of A (as column vectors in the returned matrix).

    """
    u, s, vh = np.linalg.svd(A, full_matrices=False)

    if dim is None:
        tol = max(A.shape[-2:]) * np.spacing(np.max(s, axis=-1, keepdims=True))
        dim = np.sum(s > tol, axis=-1, dtype=int)
        if not np.all(dim == dim.flat[0]):
            raise ValueError(
                "Cannot calculate the image spaces of matrices when the spaces have different dimensions."
            )
        dim = dim.flat[0]

    Q = u[..., :dim]
    return Q


def matmul(
    a,
    b,
    transpose_a=False,
    transpose_b=False,
    adjoint_a=False,
    adjoint_b=False,
    **kwargs
):
    """Matrix product of two arrays.

    Parameters
    ----------
    a, b : array_like
        Input arrays, scalars not allowed.
    transpose_a : bool
        If true, `a` is transposed before multiplication.
    transpose_b : bool
        If true, `b` is transposed before multiplication.
    adjoint_a : bool
        If true, `a` is conjugated and transposed before multiplication.
    adjoint_b : bool
        If true, `b` is conjugated and transposed before multiplication.
    **kwargs
        Additional keyword arguments for `numpy.matmul`.

    Returns
    -------
    numpy.ndarray
        The matrix product of the inputs.

    """
    if adjoint_a:
        a = np.conjugate(a)
        transpose_a = True
    if adjoint_b:
        b = np.conjugate(b)
        transpose_b = True
    if transpose_a:
        a = np.swapaxes(a, -1, -2)
    if transpose_b:
        b = np.swapaxes(b, -1, -2)
    return np.matmul(a, b, **kwargs)


def matvec(a, b, transpose_a=False, adjoint_a=False, **kwargs):
    """Matrix-vector product of two arrays.

    Parameters
    ----------
    a, b : array_like
        Input arrays, scalars not allowed.
    transpose_a : bool
        If true, `a` is transposed before multiplication.
    adjoint_a : bool
        If true, `a` is conjugated and transposed before multiplication.
    **kwargs
        Additional keyword arguments for `numpy.matmul`.

    Returns
    -------
    numpy.ndarray
        The matrix-vector product of the inputs.

    """
    result = matmul(
        a,
        np.expand_dims(b, axis=-1),
        transpose_a=transpose_a,
        adjoint_a=adjoint_a,
        **kwargs
    )
    return np.squeeze(result, axis=-1)


def roots(p):
    # TODO: only use this for len(p) > 4
    if len(p) != 4:
        return np.roots(p)

    a, b, c, d = p

    if a == 0 and b == 0:  # Linear equation
        return [-d / c]

    elif a == 0:  # Quadratic equation
        D = c ** 2 - 4 * b * d
        D = csqrt(D)
        x1 = (-c + D) / (2 * b)
        x2 = (-c - D) / (2 * b)
        return [x1, x2]

    f = ((3 * c / a) - ((b ** 2) / (a ** 2))) / 3
    g = (((2 * (b ** 3)) / (a ** 3)) - ((9 * b * c) / (a ** 2)) + (27 * d / a)) / 27
    h = (g ** 2) / 4 + (f ** 3) / 27

    if f == 0 and g == 0 and h == 0:  # All 3 roots are real and equal
        x = np.cbrt(d / a)
        return [x]

    elif h <= 0:  # All 3 roots are real

        i = np.sqrt(((g ** 2) / 4) - h)
        j = np.cbrt(i)
        k = np.arccos(-(g / (2 * i)))
        L = j * -1
        M = np.cos(k / 3)
        N = np.sqrt(3) * np.sin(k / 3)
        P = (b / (3 * a)) * -1

        x1 = 2 * j * np.cos(k / 3) - (b / (3 * a))
        x2 = L * (M + N) + P
        x3 = L * (M - N) + P

        return [x1, x2, x3]

    elif h > 0:  # One real root and two complex roots
        R = -(g / 2) + np.sqrt(h)
        S = np.cbrt(R)
        T = -(g / 2) - np.sqrt(h)
        U = np.cbrt(T)

        x1 = (S + U) - (b / (3 * a))
        x2 = -(S + U) / 2 - (b / (3 * a)) + (S - U) * np.sqrt(3) * 0.5j
        x3 = np.conj(x2)

        return [x1, x2, x3]
