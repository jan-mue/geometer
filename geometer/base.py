from abc import ABC
import numbers
from itertools import permutations

import numpy as np
import sympy

from .utils import is_multiple
from .exceptions import TensorComputationError


EQ_TOL_REL = 1e-15
EQ_TOL_ABS = 1e-8

_symbol_cache = []


def _symbols(n):
    if len(_symbol_cache) <= n:
        _symbol_cache.extend(sympy.symbols(["x" + str(i) for i in range(len(_symbol_cache), n)]))
    return _symbol_cache[0] if n == 1 else _symbol_cache[:n]


class Tensor(np.ndarray):
    """Subclass of numpy.ndarray that keeps track of covariant and contravariant indices.

    Parameters
    ----------
    *args
        Either a single iterable or multiple coordinate numbers.
    covariant : :obj:`bool` or :obj:`list` of :obj:`int`, optional
        If False, all indices are contravariant. If a list of indices indices is supplied, the specified indices of the
        array will be covariant indices and all others contravariant indices. By default all indices are covariant.
    **kwargs
        Additional arguments used to create the numpy array.

    Attributes
    ----------
    array : numpy.ndarray
        The underlying numpy array.

    """

    def __new__(cls, *args, **kwargs):

        if len(args) == 1:
            array = args[0]
        else:
            array = args

        if isinstance(array, Tensor) and 'covariant' not in kwargs:
            obj = np.array(array, **kwargs).view(cls)
            obj._covariant_indices = args[0]._covariant_indices
            obj._contravariant_indices = args[0]._contravariant_indices
            return obj

        covariant = kwargs.pop('covariant', True)

        obj = np.array(array, **kwargs).view(cls)

        if covariant is True:
            obj._covariant_indices = set(range(obj.ndim))
        else:
            obj._covariant_indices = set(covariant) if covariant else set()

        obj._contravariant_indices = set(range(obj.ndim)) - obj._covariant_indices

        return obj

    def __array_finalize__(self, obj):
        # From an explicit constructor
        if obj is None: return

        # From view casting or new-from-template
        # TODO: handle changing dimensions
        self._covariant_indices = getattr(obj, '_covariant_indices', set(range(self.ndim)))
        self._contravariant_indices = set(range(self.ndim)) - self._covariant_indices

    _HANDLED_TYPES = (np.ndarray, numbers.Number, list)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.get('out', ())
        for x in inputs + out:
            # Only support operations with instances of _HANDLED_TYPES.
            # Use ArrayLike instead of type(self) for isinstance to
            # allow subclasses that don't override __array_ufunc__ to
            # handle Tensor objects.
            if not isinstance(x, self._HANDLED_TYPES + (Tensor,)):
                return NotImplemented

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(x.array if isinstance(x, Tensor) else x for x in inputs)
        if out:
            kwargs['out'] = tuple(x.array if isinstance(x, Tensor) else x for x in out)
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if type(result) is tuple:
            # multiple return values
            return tuple(Tensor(x) for x in result)
        elif method == 'at':
            # no return value
            return None
        else:
            # one return value
            return Tensor(result, covariant=self._covariant_indices)

    @property
    def array(self):
        return self.view(np.ndarray)

    @property
    def tensor_shape(self):
        """:obj:`tuple` of :obj:`int`: The shape or type of the tensor, the first number is the number of
        covariant indices, the second the number of contravariant indices."""
        return len(self._covariant_indices), len(self._contravariant_indices)

    def tensor_product(self, other):
        """Return a new tensor that is the tensor product of this and the other tensor.

        This method will also reorder the indices of the resulting tensor, to ensure that covariant indices are in
        front of the contravariant indices.

        Parameters
        ----------
        other : Tensor
            The other tensor.

        Returns
        -------
        Tensor
            The tensor product.

        """
        offset = self.ndim
        covariant = list(self._covariant_indices) + [offset + i for i in other._covariant_indices]
        contravariant = list(self._contravariant_indices) + [offset + i for i in other._contravariant_indices]

        result = np.tensordot(self, other, 0)
        result = np.transpose(result, axes=covariant + contravariant)
        return Tensor(result, covariant=range(len(covariant)))

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = reversed(range(self.ndim))
        elif len(axes) == 1:
            axes = axes[0]

        axes = list(axes)

        if len(axes) < self.ndim:
            a = list(range(self.ndim))
            for ind in range(len(axes)):
                i, j = axes[ind], axes[(ind + 1) % len(axes)]
                a[i] = j
            axes = a

        covariant = []
        for i, j in enumerate(axes):
            if j in self._covariant_indices:
                covariant.append(i)

        return Tensor(super(Tensor, self).transpose(axes), covariant=covariant)

    @property
    def T(self):
        return self.transpose()

    def reshape(self, *args):
        return self.array.reshape(*args)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, str(self.tolist()))

    def __matmul__(self, other):
        if isinstance(other, Tensor):
            return TensorDiagram((other, self)).calculate()
        return super(Tensor, self).__matmul__(other)

    def __rmatmul__(self, other):
        if isinstance(other, Tensor):
            return TensorDiagram((self, other)).calculate()
        return super(Tensor, self).__rmatmul__(other)

    def __eq__(self, other):
        return np.allclose(self, other, rtol=EQ_TOL_REL, atol=EQ_TOL_ABS)

    def __ne__(self, other):
        return not self == other


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

    def __new__(cls, size, covariant=True):
        if size in cls._cache:
            array = cls._cache[size]
        else:
            i, j = np.triu_indices(size, 1)
            indices = np.array(list(permutations(range(size))), dtype=int).T
            diff = indices[j] - indices[i]
            diff = np.sign(diff, dtype=np.int8)
            array = np.zeros(size * [size], dtype=np.int8)
            array[tuple(indices)] = np.prod(diff, axis=0)

        cls._cache[size] = array
        return super(LeviCivitaTensor, cls).__new__(cls, array, covariant=bool(covariant))


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

    def __new__(cls, n, p=1):
        if p == 1:
            array = np.eye(n, dtype=int)
        elif (p, n) in cls._cache:
            array = cls._cache[(p, n)]
        elif p == n:
            e = LeviCivitaTensor(n)
            array = np.tensordot(e, e, 0)

            cls._cache[(p, n)] = array
        else:
            d1 = KroneckerDelta(n)
            d2 = KroneckerDelta(n, p-1)

            def calc(*args):
                return sum((-1)**(p+k+1)*d1[args[k], args[-1]]*d2[tuple(x for i, x in enumerate(args[:-1]) if i != k)] for k in range(p))

            f = np.vectorize(calc)
            array = np.fromfunction(f, tuple(2*p*[n]), dtype=int)

        return super(KroneckerDelta, cls).__new__(cls, array, covariant=range(p))


class TensorDiagram:
    """A class used to specify and calculate tensor diagrams (also called Penrose Graphical Notation).

    Each edge in the diagram represents a contraction of two indices of the tensors connected by that edge. In
    Einstein-notation that would mean that an edge from tensor A to tensor B is equivalent to the expression
    :math:`A_{i j}B^{i k}_l`, where :math:`j, k, l` are free indices. The indices to contract are chosen from front to
    back from contravariant and covariant indices of the tensors that are connected by an edge.

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

    def add_node(self, node):
        """Add a node to the tensor diagram without adding an edge/contraction.

        A diagram of nodes where none are connected is equivalent to calculating the tensor product with the
        method :meth:`Tensor.tensor_product`.

        Parameters
        ----------
        node : Tensor
            The node to add.

        """
        self._nodes.append(node)
        self._node_positions.append(self._index_count)
        self._index_count += node.ndim
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
                free_source = self.add_node(source)[0]

            if target_index is None:
                target_index = len(self._nodes)
                free_target = self.add_node(target)[1]

        if len(free_source) == 0 or len(free_target) == 0:
            raise TensorComputationError("Could not add the edge because no free indices are left.")

        # Third step: Pick some free indices
        i = free_source.pop(0)
        j = free_target.pop(0)

        if source.shape[i] != target.shape[j]:
            raise TensorComputationError(
                "Dimension of tensors is inconsistent, encountered dimensions {} and {}.".format(
                    str(source.shape[i]), str(target.shape[j])))

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
            args.append(node)
            s = slice(offset, self._node_positions[i + 1] if i + 1 < len(self._node_positions) else None)
            args.append(indices[s])

        temp = np.einsum(*args, result_indices[0] + result_indices[1])

        return Tensor(temp, covariant=range(len(result_indices[0])))

    def copy(self):
        result = TensorDiagram()
        result._nodes = self._nodes.copy()
        result._free_indices = self._free_indices.copy()
        result._node_positions = self._node_positions.copy()
        result._contraction_list = self._contraction_list.copy()
        result._index_count = self._index_count
        return result

    def __copy__(self):
        return self.copy()


class ProjectiveElement(Tensor, ABC):
    """Base class for all projective tensors, i.e. all objects that identify scalar multiples.

    """

    def __eq__(self, other):
        if isinstance(other, Tensor):

            if self.shape != other.shape:
                return False

            return is_multiple(self, other, rtol=EQ_TOL_REL, atol=EQ_TOL_ABS)

        return super(ProjectiveElement, self).__eq__(other)

    @property
    def dim(self):
        """int: The dimension of the tensor."""
        return self.shape[0] - 1
