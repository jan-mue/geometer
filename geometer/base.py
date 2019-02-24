from abc import ABC
from itertools import permutations

import numpy as np
import sympy

from .exceptions import TensorComputationError


_symbol_cache = []


def _symbols(n):
    if len(_symbol_cache) <= n:
        _symbol_cache.extend(sympy.symbols(["x" + str(i) for i in range(len(_symbol_cache), n)]))
    return _symbol_cache[0] if n == 1 else _symbol_cache[:n]


class Tensor:
    """Wrapper class around a numpy array that keeps track of covariant and contravariant indices.

    Parameters
    ----------
    *args
        Either a single iterable or multiple coordinate numbers.
    covariant : :obj:`bool` or :obj:`list` of :obj:`int`, optional
        If False, all indices are contravariant. If a list of indices indices is supplied, the specified indices of the
        array will be covariant indices and all others contravariant indices. By default all indices are covariant.

    Attributes
    ----------
    array : numpy.ndarray
        The underlying numpy array.

    """

    def __init__(self, *args, covariant=True):
        if len(args) == 1:
            if isinstance(args[0], Tensor):
                self.array = args[0].array
                self._covariant_indices = args[0]._covariant_indices
                self._contravariant_indices = args[0]._contravariant_indices
                return
            else:
                self.array = np.array(args[0])
        else:
            self.array = np.array(args)

        if covariant is True:
            self._covariant_indices = set(range(len(self.array.shape)))
        else:
            self._covariant_indices = set(covariant) if covariant else set()

        self._contravariant_indices = set(range(len(self.array.shape))) - self._covariant_indices

    @property
    def tensor_shape(self):
        """:obj:`tuple` of :obj:`int`: The shape of the indices of the tensor, the first number is the number of covariant
        indices, the second the number of contravariant indices."""
        return len(self._covariant_indices), len(self._contravariant_indices)

    def copy(self):
        return type(self)(self)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, str(self.array.tolist()))

    def __copy__(self):
        return self.copy()

    def __mul__(self, other):
        d = TensorDiagram((other, self))
        return d.calculate()

    def __rmul__(self, other):
        d = TensorDiagram((self, other))
        return d.calculate()

    def __eq__(self, other):
        if np.isscalar(other):
            return np.allclose(self.array, np.full(self.array.shape, other))
        return np.allclose(self.array, other.array)


class LeviCivitaTensor(Tensor):
    """This class can be used to construct a tensor representing the Levi-Civita symbol.

    Parameters
    ----------
    size : int
        The number of indices of the tensor.
    covariant: :obj:`bool`, optional
        If true, the tensor will only have covariant indices. Default: True

    """

    _cache = {}

    def __init__(self, size, covariant=True):
        if size in self._cache:
            array = self._cache[size]
        else:
            i, j = np.triu_indices(size, 1)
            indices = np.array(list(permutations(range(size))), dtype=int).T
            diff = indices[j] - indices[i]
            diff = np.sign(diff, dtype=np.int8)
            array = np.zeros(size * [size], dtype=np.int8)
            array[tuple(indices)] = np.prod(diff, axis=0)

            self._cache[size] = array
        super(LeviCivitaTensor, self).__init__(array, covariant=bool(covariant))


class TensorDiagram:
    """A class used to specify and calculate tensor diagrams (also called Penrose Graphical Notation).

    Parameters
    ----------
    *edges
        Variable number of tuples, that represent the edge from one tensor to another.

    """

    def __init__(self, *edges):
        self._nodes = []
        self._indices = []
        self._free_indices = []
        self._split_indices = []
        for e in edges or []:
            self.add_edge(*e)

    def add_edge(self, source, target):
        """Add an edge to the diagram.

        Parameters
        ----------
        source : Tensor
            The source tensor of the edge in the diagram.
        target : Tensor
            The target tensor of the edge in the diagram.

        """
        source_index, target_index = None, None
        index_count = 0

        def add(node):
            nonlocal index_count
            self._split_indices.append(index_count)
            self._nodes.append(node)
            ind = (node._covariant_indices.copy(), node._contravariant_indices.copy())
            self._free_indices.append(ind)
            index_count += len(node.array.shape)
            return ind

        if source.array.shape[0] != target.array.shape[0]:
            raise TensorComputationError(
                "Dimension of tensors is inconsistent, encountered dimensions {} and {}.".format(
                    str(source.array.shape[0]), str(target.array.shape[0])))

        for i, tensor in enumerate(self._nodes):
            if tensor is source:
                source_index = index_count
                free_source = self._free_indices[i][0]
            if tensor is target:
                target_index = index_count
                free_target = self._free_indices[i][1]

            index_count += len(tensor.array.shape)

            if source_index is not None and target_index is not None:
                break

        else:
            if source_index is None:
                source_index = index_count
                free_source = add(source)[0]
            if target_index is None:
                target_index = index_count
                free_target = add(target)[1]

            self._indices.extend(range(len(self._indices), index_count))

        if len(free_source) == 0 or len(free_target) == 0:
            raise TensorComputationError("Could not add the edge because no free indices are left in the following tensors: " + str((source, target)))

        i = source_index + free_source.pop()
        j = target_index + free_target.pop()
        self._indices[max(i, j)] = min(i, j)

    def calculate(self):
        """Calculates the result of the diagram.

        Returns
        -------
        Tensor
            The tensor resulting from the specified tensor diagram.

        """
        args = []
        result_indices = ([], [])
        for i, (node, ind, offset) in enumerate(zip(self._nodes, self._free_indices, self._split_indices)):
            result_indices[0].extend(offset + x for x in ind[0])
            result_indices[1].extend(offset + x for x in ind[1])
            args.append(node.array)
            s = slice(offset, self._split_indices[i + 1] if i + 1 < len(self._split_indices) else None)
            args.append(self._indices[s])

        cov_count = len(result_indices[0])
        result_indices = result_indices[0] + result_indices[1]

        x = np.einsum(*args, result_indices)
        return Tensor(x, covariant=range(cov_count))


class ProjectiveElement(Tensor, ABC):
    """Base class for all projective tensors, i.e. all objects that identify scalar multiples.

    """

    def __eq__(self, other):
        if isinstance(other, Tensor):

            if self.array.shape != other.array.shape:
                return False

            # By Cauchy-Schwarz |(x,y)| = ||x||*||y|| iff x = cy
            a = self.array.ravel()
            b = other.array.ravel()
            return np.isclose(np.abs(np.vdot(a, b)) ** 2, np.vdot(a, a) * np.vdot(b, b))
        return NotImplemented

    @property
    def dim(self):
        """int: The dimension of the tensor."""
        return self.array.shape[0] - 1
