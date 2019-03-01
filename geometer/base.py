from abc import ABC, abstractmethod
import warnings
from itertools import permutations

import numpy as np
import sympy

from .exceptions import TensorComputationError


_symbol_cache = []


def _symbols(n):
    if len(_symbol_cache) <= n:
        _symbol_cache.extend(sympy.symbols(["x" + str(i) for i in range(len(_symbol_cache), n)]))
    return _symbol_cache[0] if n == 1 else _symbol_cache[:n]


class Shape(ABC):
    """Base class for geometric shapes, i.e. line segments, polytopes and polygons.

    Parameters
    ----------
    *args
        The sides of the shape.

    """

    def __init__(self, *args):
        self._sides = list(args)

    @property
    def dim(self):
        """int: The dimension of the space that the shape lives in."""
        return self.sides[0].dim

    @property
    @abstractmethod
    def vertices(self):
        """list of Points: The vertices of the shape."""
        pass

    @property
    def sides(self):
        """list of Shape: The sides of the shape."""
        return self._sides

    @abstractmethod
    def intersect(self, other):
        """Intersect the shape with another object.

        Parameters
        ----------
        other : Line, Plane, Quadric or Shape

        Returns
        -------
        list of Point
            The points of intersection.

        """
        pass

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, ", ".join(str(v) for v in self.vertices))


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
        if len(args) == 0:
            raise TypeError("At least one argument is required.")

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
            self._covariant_indices = set(range(self.array.ndim))
        else:
            self._covariant_indices = set(covariant) if covariant else set()

        self._contravariant_indices = set(range(self.array.ndim)) - self._covariant_indices

    @property
    def tensor_shape(self):
        """:obj:`tuple` of :obj:`int`: The shape of the indices of the tensor, the first number is the number of
        covariant indices, the second the number of contravariant indices."""
        return len(self._covariant_indices), len(self._contravariant_indices)

    def tensordot(self, other):
        """Return a new tensor that is the tensor product of this and the other tensor.

        Parameters
        ----------
        other : Tensor
            The other tensor.

        Returns
        -------
        Tensor
            The tensor product.

        """
        ind = self._covariant_indices
        d = self.array.ndim
        ind.update(d + i for i in other._covariant_indices)
        return Tensor(np.tensordot(self.array, other.array, 0), covariant=ind)

    def copy(self):
        return type(self)(self)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, str(self.array.tolist()))

    def __copy__(self):
        return self.copy()

    def __mul__(self, other):
        return TensorDiagram((other, self))

    def __rmul__(self, other):
        return TensorDiagram((self, other))

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


class TensorDiagram(Tensor):
    """A class used to specify and calculate tensor diagrams (also called Penrose Graphical Notation).

    Parameters
    ----------
    *edges
        Variable number of tuples, that represent the edge from one tensor to another.

    """

    def __init__(self, *edges):
        self._nodes = []
        self._index_mapping = []

        if len(edges) == 0:
            raise TypeError("Cannot create an empty tensor diagram.")

        free_indices = []
        split = []
        contract_indices = []

        def add(node):
            nonlocal index_count
            split.append(index_count)
            self._nodes.append(node)
            ind = (node._covariant_indices.copy(), node._contravariant_indices.copy())
            free_indices.append(ind)
            index_count += node.array.ndim
            return ind

        for source, target in edges:

            if source.array.shape[0] != target.array.shape[0]:
                raise TensorComputationError("Dimension of tensors is inconsistent, encountered dimensions {} and {}.".format(str(source.array.shape[0]), str(target.array.shape[0])))

            source_index, target_index = None, None
            index_count = 0

            for i, tensor in enumerate(self._nodes):
                if tensor is source:
                    source_index = index_count
                    free_source = free_indices[i][0]
                if tensor is target:
                    target_index = index_count
                    free_target = free_indices[i][1]

                index_count += tensor.array.ndim

                if source_index is not None and target_index is not None:
                    break

            else:
                if source_index is None:
                    source_index = index_count
                    free_source = add(source)[0]
                if target_index is None:
                    target_index = index_count
                    free_target = add(target)[1]

            if len(free_source) == 0 or len(free_target) == 0:
                raise TensorComputationError("Could not add the edge because no free indices are left in the following tensors: " + str((source, target)))

            i = source_index + free_source.pop()
            j = target_index + free_target.pop()
            contract_indices.append((min(i, j), max(i, j)))

        indices = list(range(index_count))
        for i, j in contract_indices:
            indices[j] = i

        args = []
        result_indices = ([], [])
        index_mapping = ([], [])
        for i, (node, ind, offset) in enumerate(zip(self._nodes, free_indices, split)):
            index_mapping[0].extend([i] * len(ind[0]))
            index_mapping[1].extend([i] * len(ind[1]))
            result_indices[0].extend(offset + x for x in ind[0])
            result_indices[1].extend(offset + x for x in ind[1])
            args.append(node.array)
            s = slice(offset, split[i + 1] if i + 1 < len(split) else None)
            args.append(indices[s])

        cov_count = len(result_indices[0])
        result_indices = result_indices[0] + result_indices[1]

        x = np.einsum(*args, result_indices)

        self._index_mapping = index_mapping[0] + index_mapping[1]
        super(TensorDiagram, self).__init__(x, covariant=range(cov_count))

    def _contract_index(self, i, j):
        indices = list(range(self.array.ndim))
        indices[max(i, j)] = min(i, j)
        self.array = np.einsum(self.array, indices)
        self._covariant_indices.remove(i)
        self._contravariant_indices.remove(j)
        self._index_mapping.pop(i)
        self._index_mapping.pop(j)

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
        for i, node in enumerate(self._nodes):
            if node is source:
                source_index = i
            if node is target:
                target_index = i

            if source_index is not None and target_index is not None:
                break

        def add(tensor):
            # TODO: directly use tensordot to contract
            self.array = self.tensordot(tensor).array
            self._index_mapping.extend([len(self._nodes)] * tensor.array.ndim)
            self._nodes.append(tensor)

        if target_index is None:
            free_target = set(sum(self.tensor_shape) + x for x in target._contravariant_indices)
            add(target)
        else:
            free_target = set()
            for i in self._contravariant_indices:
                if self._index_mapping[i] == target_index:
                    free_target.add(i)

        if source_index is None:
            free_source = set(sum(self.tensor_shape) + x for x in source._covariant_indices)
            add(source)
        else:
            free_source = set()
            for i in self._covariant_indices:
                if self._index_mapping[i] == source_index:
                    free_source.add(i)

        if len(free_source) == 0 or len(free_target) == 0:
            raise TensorComputationError("Could not add the edge because no free indices are left.")

        self._contract_index(free_source.pop(), free_target.pop())

    def calculate(self):
        """Calculates the result of the diagram.

        Deprecated as of version v0.2.

        Returns
        -------
        Tensor
            The tensor resulting from the specified tensor diagram.

        """
        warnings.warn("deprecated", DeprecationWarning)
        return self


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
