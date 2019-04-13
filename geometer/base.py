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
                self.array = np.atleast_1d(args[0])
        else:
            self.array = np.array(args)

        if covariant is True:
            self._covariant_indices = set(range(self.array.ndim))
        else:
            self._covariant_indices = set(covariant) if covariant else set()

        self._contravariant_indices = set(range(self.array.ndim)) - self._covariant_indices

    @property
    def tensor_shape(self):
        """:obj:`tuple` of :obj:`int`: The shape or type of the tensor, the first number is the number of
        covariant indices, the second the number of contravariant indices."""
        return len(self._covariant_indices), len(self._contravariant_indices)

    def tensor_product(self, other):
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
        d = self.array.ndim
        covariant = self._covariant_indices.copy()
        covariant.update(d + i for i in other._covariant_indices)
        return Tensor(np.tensordot(self.array, other.array, 0), covariant=covariant)

    def transpose(self, perm=None):
        """Permute the indices of the tensor.

        Parameters
        ----------
        perm : tuple of int, optional
            A list of permuted indices or a shorter list representing a permutation in cycle notation.
            By default, the indices are reversed.

        Returns
        -------
        Tensor
            The tensor with permuted indices.

        """
        if perm is None:
            perm = reversed(range(self.array.ndim))

        perm = list(perm)

        if len(perm) < self.array.ndim:
            a = list(range(self.array.ndim))
            for ind in range(len(perm)):
                i, j = perm[ind], perm[(ind + 1) % len(perm)]
                a[i] = j
            perm = a

        covariant = []
        for i, j in enumerate(perm):
            if i in self._covariant_indices:
                covariant.append(j)

        return Tensor(np.transpose(self.array, perm), covariant=covariant)

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

    def __add__(self, other):
        return Tensor(self.array + other.array, covariant=self._covariant_indices)

    def __sub__(self, other):
        return Tensor(self.array - other.array, covariant=self._covariant_indices)

    def __eq__(self, other):
        if isinstance(other, Tensor):
            other = other.array
        return np.allclose(self.array, other)


class LeviCivitaTensor(Tensor):
    """This class can be used to construct a tensor representing the Levi-Civita symbol.

    Parameters
    ----------
    size : int
        The number of indices of the tensor.
    covariant: :obj:`bool`, optional
        If true, the tensor will only have covariant indices. Default: True

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Levi-Civita_symbol#Generalization_to_n_dimensions

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


class KroneckerDelta(Tensor):
    """This class can be used to construct a (p, p)-tensor representing the Kronecker delta tensor.

    Parameters
    ----------
    n : int
        The dimension of the tensor.
    p : int, optional
        The number of covariant and contravariant indices of the tensor, default is 1.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Kronecker_delta#Generalizations

    """

    _cache = {}

    def __init__(self, n, p=1):
        if p == 1:
            array = np.eye(n)
        elif (p, n) in self._cache:
            array = self._cache[(p, n)]
        elif p == n:
            e = LeviCivitaTensor(n)
            array = np.tensordot(e.array, e.array, 0)

            self._cache[(p, n)] = array
        else:
            d1 = KroneckerDelta(n)
            d2 = KroneckerDelta(n, p-1)

            def calc(*args):
                return sum((-1)**(p+k+1)*d1.array[args[k], args[-1]]*d2.array[tuple(x for i, x in enumerate(args[:-1]) if i != k)] for k in range(p))

            f = np.vectorize(calc)
            array = np.fromfunction(f, tuple(2*p*[n]), dtype=int)

        super(KroneckerDelta, self).__init__(array, covariant=range(p))


class TensorDiagram:
    """A class used to specify and calculate tensor diagrams (also called Penrose Graphical Notation).

    Each edge in the diagram represents a contraction of two indices of the tensors connected by that edge. In
    Einstein-notation that would mean that an edge from tensor A to tensor B is equivalent to the expression
    :math:`A_{i j}B^{k j}_l`, where :math:`i, k, l` are free indices. The indices to contract are chosen from back to
    front from contravariant and covariant indices of the tensors that are connected by an edge.

    Parameters
    ----------
    *edges
        Variable number of tuples, that represent the edge from one tensor to another.


    References
    ----------
    .. [1] https://www-m10.ma.tum.de/foswiki/pub/Lehrstuhl/PublikationenJRG/52_TensorDiagrams.pdf
    .. [2] J. Richter-Gebert: Perspectives on Projective Geometry, Chapters 13-14

    """

    def __init__(self, *edges):
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
            raise TensorComputationError(
                "Dimension of tensors is inconsistent, encountered dimensions {} and {}.".format(
                    str(source.array.shape[0]), str(target.array.shape[0])))

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
            raise TensorComputationError("Could not add the edge because no free indices are left.")

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

        temp = np.einsum(*args, result_indices[0] + result_indices[1])

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
            ab = np.vdot(a, b)
            return np.isclose(ab * np.conjugate(ab), np.vdot(a, a) * np.vdot(b, b))
        return NotImplemented

    @property
    def dim(self):
        """int: The dimension of the tensor."""
        return self.array.shape[0] - 1
