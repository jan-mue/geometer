import math
import numpy as np


def is_multiple(a, b, axis=None, rtol=1.e-15, atol=1.e-8):
    """Returns a boolean array where two arrays are scalar multiples of each other along a given axis.

    This function compares the absolute value of the scalar product and the product of the norm of the arrays (along
    an axis). The Cauchy-Schwarz inequality guarantees in its edge case that this equality holds if and only if one
    of the vectors is a scalar multiple of the other.

    For documentation of the tolerance parameters see :func:`numpy.isclose`.

    Parameters
    ----------
    a, b : array_like
        Input arrays to compare.
    axis : None or int
        The axis along which the two arrays are compared.
        The default axis=None will compare the whole arrays and return only a single boolean value.
    rtol : float, optional
        The relative tolerance parameter.
    atol : float, optional
        The absolute tolerance parameter.

    Returns
    -------
    numpy.ndarray or bool
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
    return np.isclose(ab*ab.conj(), np.sum(a*a.conj(), axis=axis)*np.sum(b*b.conj(), axis=axis), rtol, atol)


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
        x = np.array(args[0])
    else:
        x = np.array(args)

    n = int(1+np.sqrt(1+8*len(x)))//2

    if n == 3:
        a, b, c = x
        return np.array([[0, c, -b],
                         [-c, 0, a],
                         [b, -a, 0]])

    result = np.zeros((n, n), x.dtype)
    i, j = np.triu_indices(n, 1)
    i, j = i[::-1], j[::-1]
    result[j, i] = -x
    result[i, j] = x

    return result


def adjugate(A):
    r"""Calculates the adjugate matrix using tensor diagrams.

    This function uses the following formula for the adjugate matrix (in Einstein notation):

    .. math::
        \textrm{adj}(A)_{ij} = \frac{1}{(n-1)!} \varepsilon_{i\ i_2 \ldots i_n} \varepsilon_{j\ j_2 \ldots j_n} A_{j_2 i_2} \ldots A_{j_n i_n}

    Source (German):
    https://de.wikipedia.org/wiki/Levi-Civita-Symbol#Zusammenhang_mit_der_Determinante

    Parameters
    ----------
    A : array_like
        A square matrix.

    Returns
    -------
    numpy.ndarray
        The adjugate of A.

    """
    from ..base import TensorDiagram, Tensor, LeviCivitaTensor

    A = np.array(A)
    n = A.shape[0]

    e1 = LeviCivitaTensor(n, False)
    e2 = LeviCivitaTensor(n, False)
    tensors = [Tensor(A) for _ in range(n-1)]
    diagram = TensorDiagram(*[(t, e1) for t in tensors], *[(t, e2) for t in tensors])

    return diagram.calculate().array.T / math.factorial(n-1)


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
    tol = max(A.shape) * np.spacing(np.max(s))
    dim = np.sum(s > tol, dtype=int)
    Q = vh[dim:, :].T.conj()
    return Q


def orth(A):
    """Constructs an orthonormal basis for the range of A using SVD.

    Parameters
    ----------
    A : ndarray
        The input matrix.

    Returns
    -------
    numpy.ndarray
        Orthonormal basis for the range of A (as column vectors in the returned matrix).

    """
    u, s, vh = np.linalg.svd(A, full_matrices=False)
    tol = max(A.shape) * np.spacing(np.max(s))
    dim = np.sum(s > tol, dtype=int)
    Q = u[:, :dim]
    return Q
