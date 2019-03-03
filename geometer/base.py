from abc import ABC, abstractmethod
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
            self._covariant_indices = set(range(len(self.array.shape)))
        else:
            self._covariant_indices = set(covariant) if covariant else set()

        self._contravariant_indices = set(range(len(self.array.shape))) - self._covariant_indices

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
        d = len(self.array.shape)
        ind.update(d + i for i in other._covariant_indices)
        return Tensor(np.tensordot(self.array, other.array, 0), covariant=ind)

    def copy(self):
        return type(self)(self)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, str(self.array.tolist()))

    def __copy__(self):
        return self.copy()

    def __mul__(self, other):
        return TensorDiagram((other, self)).calculate()

    def __rmul__(self, other):
        return TensorDiagram((self, other)).calculate()

    def __eq__(self, other):
        a = self.array
        if np.isscalar(other):
            b = np.full(self.array.shape, other)
        elif isinstance(other, Tensor):
            b = other.array
        else:
            return NotImplemented
        try:
            return np.allclose(a, b)
        except TypeError:
            return np.all(a == b)


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

        if len(edges) == 0:
            raise TypeError("Cannot create an empty tensor diagram.")

        self._nodes = []
        self._free_indices = []
        self._node_positions = []
        self._contraction_list = []
        self._index_count = 0

        for e in edges:
            self.add_edge(*e)

    def _add_node(self, node):
        self._nodes.append(node)
        self._node_positions.append(self._index_count)
        self._index_count += node.array.ndim
        ind = (list(node._covariant_indices), list(node._contravariant_indices))
        self._free_indices.append(ind)
        return ind

    def add_edge(self, source, target):
        """Add an edge to the diagram.

        Parameters
        ----------
        source : Tensor
            The source tensor of the edge in the diagram.
        target : Tensor
            The target tensor of the edge in the diagram.

        """

        if source.array.shape[0] != target.array.shape[0]:
            raise TensorComputationError("Dimension of tensors is inconsistent, encountered dimensions {} and {}.".format(str(source.array.shape[0]), str(target.array.shape[0])))

        # First step: Find nodes if they are already in the diagram
        source_index, target_index = None, None
        for index, node in enumerate(self._nodes):
            if node is source:
                source_index = index
                free_source = self._free_indices[index][0]
            if node is target:
                target_index = index
                free_target = self._free_indices[index][1]

            if source_index is not None and target_index is not None:
                break

        # One or both nodes were not found in the diagram
        else:
            # Second step: Add new nodes to nodes and free indices list
            if source_index is None:
                source_index = len(self._nodes)
                free_source = self._add_node(source)[0]

            if target_index is None:
                target_index = len(self._nodes)
                free_target = self._add_node(target)[1]

        if len(free_source) == 0 or len(free_target) == 0:
            raise TensorComputationError("Could not add the edge because no free indices are left in the following tensors: " + str((source, target)))

        # Third step: Pick some free indices
        i = free_source.pop()
        j = free_target.pop()

        self._contraction_list.append((source_index, target_index, i, j))

    def calculate(self):
        """Calculates the result of the diagram.

        Returns
        -------
        Tensor
            The tensor resulting from the specified tensor diagram.

        """
        # Build the list of indices for einsum
        indices = list(range(self._index_count))
        for source_index, target_index, i, j in self._contraction_list:
            i = self._node_positions[source_index] + i
            j = self._node_positions[target_index] + j
            indices[max(i, j)] = min(i, j)

        # Split the indices and insert the arrays
        args = []
        result_indices = ([], [])
        for i, (node, ind, offset) in enumerate(zip(self._nodes, self._free_indices, self._node_positions)):
            result_indices[0].extend(offset + x for x in ind[0])
            result_indices[1].extend(offset + x for x in ind[1])
            args.append(node.array)
            s = slice(offset, self._node_positions[i + 1] if i + 1 < len(self._node_positions) else None)
            args.append(indices[s])

        try:
            # This only works for numeric types build into numpy (because c_einsum is called)
            temp = np.einsum(*args, result_indices[0] + result_indices[1])

        except TypeError:
            # einsum failed -> calculate contraction step by step
            temp = None
            operands = [node.array for node in self._nodes]
            contracted_nodes = set()
            for source_index, target_index, i, j in self._contraction_list:
                if source_index in contracted_nodes and target_index in contracted_nodes:
                    i += self._node_positions[source_index]
                    j += self._node_positions[target_index]

                    # Sum over the axes i and j where they are equal
                    temp = np.sum(np.diagonal(temp, axis1=i, axis2=j), axis=-1)

                elif source_index in contracted_nodes:
                    i += self._node_positions[source_index]

                    # Multiply and sum over axes i and j and add the remaining axes of target to the end of temp
                    temp = np.tensordot(temp, operands[target_index], axes=(i, j))
                    contracted_nodes.add(target_index)

                elif target_index in contracted_nodes:
                    j += self._node_positions[target_index]

                    # Same as in the last case, tensordot with source
                    temp = np.tensordot(temp, operands[source_index], axes=(j, i))
                    contracted_nodes.add(source_index)

                else:
                    x = np.tensordot(operands[source_index], operands[target_index], axes=(i, j))

                    # the indices of x are not contracted with any indices of temp -> use tensor product
                    temp = x if temp is None else np.tensordot(temp, x, 0)
                    contracted_nodes.update((source_index, target_index))

            # To bring the contracted axes in the right order (covariant in front), build index list
            index_count = 0
            result_indices = ([], [])
            for ind in self._free_indices:
                result_indices[0].extend(range(index_count, index_count + len(ind[0])))
                index_count += len(ind[0])
                result_indices[1].extend(range(index_count, index_count + len(ind[1])))
                index_count += len(ind[1])

            # Reorder the axes
            temp = np.transpose(temp, axes=tuple(result_indices[0] + result_indices[1]))

        return Tensor(temp, covariant=range(len(result_indices[0])))


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
