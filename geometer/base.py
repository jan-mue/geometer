from abc import ABC
import warnings
from itertools import permutations

import numpy as np
from .exceptions import TensorComputationError


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
                covariant = args[0]._covariant_indices
            else:
                self.array = np.atleast_1d(args[0])
        else:
            self.array = np.array(args)

        if covariant is True:
            covariant = range(len(self.array.shape))
        elif not covariant:
            covariant = []

        self._covariant_indices = set(covariant)
        self._contravariant_indices = set(range(len(self.array.shape))) - self._covariant_indices

    @property
    def tensor_shape(self):
        """:obj:`tuple` of :obj:`int`: The shape of the indices of the tensor, the first number is the number of covariant
        indices, the second the number of contravariant indices."""
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
        d = len(self.array.shape)
        ind.update(d + i for i in other._covariant_indices)
        return Tensor(np.tensordot(self.array, other.array, 0), covariant=ind)

    def copy(self):
        return type(self)(self)

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

        indices = []
        free_indices = []
        split = []

        def add(a, b):
            result = [None, None]
            index_count = 0
            not_found = [(0, a), (1, b)]

            for i, tensor in enumerate(self._nodes):
                for j, node in not_found:
                    if tensor is node:
                        not_found.remove((j, node))
                        result[j] = (free_indices[i], index_count)

                index_count += len(tensor.array.shape)

                if len(not_found) == 0:
                    break

            else:
                for j, node in not_found:
                    split.append(index_count)
                    self._nodes.append(node)
                    ind = {
                        "covariant": node._covariant_indices.copy(),
                        "contravariant": node._contravariant_indices.copy()
                    }
                    free_indices.append(ind)
                    result[j] = (ind, index_count)
                    index_count += len(node.array.shape)

                indices.extend(range(len(indices), index_count))

            return result

        for edge in edges:
            (free_source, source_index), (free_target, target_index) = add(*edge)

            if len(free_source["covariant"]) == 0 or len(free_target["contravariant"]) == 0:
                raise TensorComputationError("Could not add the edge because no free indices are left.")

            i = source_index + free_source["covariant"].pop()
            j = target_index + free_target["contravariant"].pop()
            indices[max(i, j)] = min(i, j)

        indices = np.split(indices, split[1:])
        args = [x for i, node in enumerate(self._nodes) for x in (node.array, indices[i].tolist())]

        result_indices = {
            "covariant": [],
            "contravariant": []
        }
        index_mapping = [[], []]
        for i, (offset, ind) in enumerate(zip(split, free_indices)):
            index_mapping[0].extend([i]*len(ind["covariant"]))
            index_mapping[1].extend([i]*len(ind["contravariant"]))
            result_indices["covariant"].extend(offset + x for x in ind["covariant"])
            result_indices["contravariant"].extend(offset + x for x in ind["contravariant"])
        cov_count = len(result_indices["covariant"])
        result_indices = result_indices["covariant"] + result_indices["contravariant"]

        x = np.einsum(*args, result_indices)

        self._index_mapping = index_mapping[0] + index_mapping[1]
        super(TensorDiagram, self).__init__(x, covariant=range(cov_count))

    def _contract_index(self, i, j):
        indices = list(range(len(self.array.shape)))
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
            elif node is target:
                target_index = i

            if source_index is not None and target_index is not None:
                break

        def add(tensor):
            self.array = self.tensordot(tensor).array
            self._index_mapping.extend([len(self._nodes)] * len(tensor.array.shape))
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
        # By Cauchy-Schwarz |(x,y)| = ||x||*||y|| iff x = cy
        a = self.array.ravel()
        b = other.array.ravel()
        return np.isclose(np.abs(np.vdot(a, b)) ** 2, np.vdot(a, a) * np.vdot(b, b))

    def __len__(self):
        return np.product(self.array.shape)

    @property
    def dim(self):
        """int: The dimension of the tensor."""
        return self.array.shape[0] - 1
