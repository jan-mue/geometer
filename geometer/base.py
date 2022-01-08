from abc import ABC
from itertools import permutations

import numpy as np

from .exceptions import TensorComputationError
from .utils import is_multiple, normalize_index, posify_index, sanitize_index

EQ_TOL_REL = 1e-15
EQ_TOL_ABS = 1e-8


class Tensor:
    """Wrapper class around a numpy array that keeps track of covariant and contravariant indices.

    Covariant indices are the lower indices (subscripts) and contravariant indices are the upper indices (superscripts)
    of a tensor (see [1]).

    Parameters
    ----------
    *args
        Either a single iterable or multiple coordinate numbers.
    covariant : :obj:`bool` or :obj:`list` of :obj:`int`, optional
        If False, all indices are contravariant. If a list of indices indices is supplied, the specified indices of the
        array will be covariant indices and all others contravariant indices. By default all indices are covariant.
    tensor_rank : int or None, optional
        If the object contains multiple tensors, this parameter specifies the rank of the tensors contained in the
        collection. By default only a single tensor is contained in a Tensor object.
    **kwargs
        Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    Attributes
    ----------
    array : numpy.ndarray
        The underlying numpy array.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Ricci_calculus#Upper_and_lower_indices

    """

    def __init__(self, *args, covariant=True, tensor_rank=None, **kwargs):
        if len(args) == 0:
            raise TypeError("At least one argument is required.")

        if len(args) == 1:
            if isinstance(args[0], Tensor):
                self.array = np.array(args[0].array, **kwargs)
                self._collection_indices = args[0]._collection_indices
                self._covariant_indices = args[0]._covariant_indices
                self._contravariant_indices = args[0]._contravariant_indices
                return
            else:
                self.array = np.array(args[0], **kwargs)
        else:
            self.array = np.array(args, **kwargs)

        if tensor_rank is None:
            tensor_rank = self.rank
        elif tensor_rank < 0:
            tensor_rank += self.shape[-1]

        if self.rank < tensor_rank:
            raise ValueError("tensor_rank must be smaller than or equal to the number of array dimensions.")

        n_col = self.rank - tensor_rank
        self._collection_indices = set(range(n_col))

        if covariant is True:
            self._covariant_indices = set(range(n_col, self.rank))
        elif covariant is False:
            self._covariant_indices = set()
        else:
            self._covariant_indices = set()
            for idx in covariant:
                if not -tensor_rank <= idx < tensor_rank:
                    raise IndexError("Index out of range")
                idx = sanitize_index(idx)
                idx = posify_index(tensor_rank, idx)
                self._covariant_indices.add(n_col + idx)

        self._contravariant_indices = set(range(self.rank)) - self._covariant_indices - self._collection_indices

    def __apply__(self, transformation):
        ts = self.tensor_shape
        edges = [(self, transformation.copy()) for _ in range(ts[0])]
        if ts[1] > 0:
            inv = transformation.inverse()
            edges.extend((inv.copy(), self) for _ in range(ts[1]))
        diagram = TensorDiagram(*edges)
        result = self.copy()
        result.array = diagram.calculate().array
        return result

    @property
    def shape(self):
        """:obj:`tuple` of :obj:`int`: The shape of the underlying numpy array, same as ``self.array.shape``."""
        return self.array.shape

    @property
    def dtype(self):
        """numpy.dtype: The dtype of the underlying numpy array, same as ``self.array.dtype``."""
        return self.array.dtype

    @property
    def tensor_shape(self):
        """:obj:`tuple` of :obj:`int`: The shape or type of the tensor, the first number is the number of
        covariant indices, the second the number of contravariant indices."""
        return len(self._covariant_indices), len(self._contravariant_indices)

    @property
    def rank(self):
        """int: The rank of the Tensor, same as ``self.array.ndim``."""
        return self.array.ndim

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
        offset = self.rank
        covariant = list(self._covariant_indices) + [offset + i for i in other._covariant_indices]
        contravariant = list(self._contravariant_indices) + [offset + i for i in other._contravariant_indices]

        result = np.tensordot(self.array, other.array, 0)
        result = np.transpose(result, axes=covariant + contravariant)
        return Tensor(result, covariant=range(len(covariant)), copy=False)

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
            perm = reversed(range(self.rank))

        perm = list(perm)

        if len(perm) < self.rank:
            a = list(range(self.rank))
            for ind in range(len(perm)):
                i, j = perm[ind], perm[(ind + 1) % len(perm)]
                a[i] = j
            perm = a

        covariant_indices = []
        contravariant_indices = []
        collection_indices = []
        for i, j in enumerate(perm):
            if j in self._covariant_indices:
                covariant_indices.append(i)
            elif j in self._contravariant_indices:
                contravariant_indices.append(i)
            elif j in self._collection_indices:
                collection_indices.append(i)

        # TODO: return Tensor
        result = self.copy()
        result.array = self.array.transpose(perm)
        result._covariant_indices = set(covariant_indices)
        result._contravariant_indices = set(contravariant_indices)
        result._collection_indices = set(collection_indices)

        return result

    @property
    def T(self):
        """The transposed tensor, same as ``self.transpose()``."""
        return self.transpose()

    def copy(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def is_zero(self, tol=EQ_TOL_ABS):
        """Test whether the tensor is zero with respect to covariant and contravariant indices.

        Parameters
        ----------
        tol : float, optional
            The accepted tolerance.

        Returns
        -------
        bool
            True if the tensor is zero. If there are more indices than the covariant and contravariant indices,
            a boolean array is returned.

        """
        axes = tuple(self._covariant_indices) + tuple(self._contravariant_indices)
        return np.all(np.isclose(self.array, 0, atol=tol), axis=axes)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, str(self.array.tolist()))

    def __getitem__(self, index):

        result = self.array[index]

        if np.isscalar(result):
            return result

        index = normalize_index(index, self.shape)

        covariant_indices = []
        contravariant_indices = []
        collection_indices = []

        advanced_indices = []

        index_mapping = list(range(self.rank))
        i = 0
        for ind in index:

            # axis with integer index will be removed
            if isinstance(ind, int):
                index_mapping.pop(i)
                continue

            # new axis inserted by None index
            if ind is None:
                index_mapping.insert(i, None)

            # advanced indexing
            elif isinstance(ind, np.ndarray):
                advanced_indices.append(i)

            i += 1

        if len(advanced_indices) > 0:

            b = np.broadcast(*[index[i] for i in advanced_indices])
            a0, a1 = advanced_indices[0], advanced_indices[-1]

            if advanced_indices != list(range(a0, a1 + 1)):
                # create advanced indices in front
                for i in advanced_indices:
                    index_mapping.remove(i)
                index_mapping = [None] * b.ndim + index_mapping
            else:
                # replace indices with broadcast shape
                index_mapping = index_mapping[:a0] + [None] * b.ndim + index_mapping[a1 + 1:]

        for new_axis, old_axis in enumerate(index_mapping):
            if old_axis is None or old_axis in self._collection_indices:
                collection_indices.append(new_axis)
            elif old_axis in self._covariant_indices:
                covariant_indices.append(new_axis)
            elif old_axis in self._contravariant_indices:
                contravariant_indices.append(new_axis)

        result = Tensor(result, copy=False)
        result._covariant_indices = set(covariant_indices)
        result._contravariant_indices = set(contravariant_indices)
        result._collection_indices = set(collection_indices)

        return result

    def __setitem__(self, key, value):
        if isinstance(value, Tensor):
            value = value.array
        self.array[key] = value

    def __copy__(self):
        return self.copy()

    def __mul__(self, other):
        if np.isscalar(other):
            return Tensor(self.array * other, covariant=self._covariant_indices, copy=False)
        if not isinstance(other, Tensor):
            other = Tensor(other, copy=False)
        return TensorDiagram((other, self)).calculate()

    def __rmul__(self, other):
        if np.isscalar(other):
            return self * other
        if not isinstance(other, Tensor):
            other = Tensor(other, copy=False)
        return TensorDiagram((self, other)).calculate()

    def __pow__(self, power, modulo=None):
        if modulo is not None or not isinstance(power, int) or power < 1:
            return NotImplemented

        if power == 1:
            return self.copy()

        d = TensorDiagram()
        prev = self
        for _ in range(power - 1):
            cur = prev.copy()
            d.add_edge(cur, prev)
            prev = cur

        return d.calculate()

    def __truediv__(self, other):
        if np.isscalar(other):
            return Tensor(self.array / other, covariant=self._covariant_indices, copy=False)
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, Tensor):
            other = other.array
        return Tensor(self.array + other, covariant=self._covariant_indices, copy=False)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, Tensor):
            other = other.array
        return Tensor(self.array - other, covariant=self._covariant_indices, copy=False)

    def __rsub__(self, other):
        return -self + other

    def __neg__(self):
        return self * (-1)

    def __eq__(self, other):
        if isinstance(other, Tensor):
            other = other.array
        return np.allclose(self.array, other, rtol=EQ_TOL_REL, atol=EQ_TOL_ABS)


class TensorCollection(Tensor):
    """A collection of Tensor objects with identical covariant/contravariant indices and shape stored in a tensor.

    Parameters
    ----------
    elements : array_like
        A (nested) sequence of Tensor objects, a numpy array, a Tensor or a (nested) sequence of numbers.
        If the sequence contains Tensor objects, they must all have the same shape and the same
        covariant/contravariant indices.
    covariant : :obj:`bool` or :obj:`list` of :obj:`int`, optional
        This parameter is used when elements is a general array or sequence of numbers, to determine the covariant
        indices of tensors contained in the collection. If False, all indices of all tensors in the collection are
        contravariant. If instead a sequence of indices is supplied, the specified indices of all tensors in the
        collection will be covariant indices and the rest of the indices of each tensor will be contravariant.
        By default all indices are covariant.
    tensor_rank : int, optional
        The rank of the tensors contained in the collection. This parameter is only used if elements is not a sequence
        of Tensor objects or a single tensor. Default is 1.
    **kwargs
        Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    """

    _element_class = Tensor

    def __init__(self, elements, *, covariant=True, tensor_rank=1, **kwargs):
        if isinstance(elements, Tensor):
            super(TensorCollection, self).__init__(elements, **kwargs)
            return

        elements = np.asarray(elements)

        # unpack tensors contained in elements
        if elements.dtype.hasobject and elements.size > 0:
            flat_elements = []
            for t in elements.flat:
                if not isinstance(t, Tensor):
                    break
                flat_elements.append(t.array)
            else:
                flat_elements = np.array(flat_elements)
                tensor_rank = t.rank
                covariant = t._covariant_indices
                elements = flat_elements.reshape(elements.shape + t.shape)
                kwargs["copy"] = False

        super(TensorCollection, self).__init__(elements, covariant=covariant, tensor_rank=tensor_rank, **kwargs)

    def expand_dims(self, axis):
        """Add a new index to the collection.

        Parameters
        ----------
        axis : int
            Position in the new shape where the new axis is placed.

        Returns
        -------
        TensorCollection
            The tensor collection with an additional index.

        """
        result = self.copy()
        result.array = np.expand_dims(self.array, axis)

        axis = sanitize_index(axis)
        axis = posify_index(self.rank, axis)
        result._collection_indices = set(i + 1 if i >= axis else i for i in self._collection_indices)
        result._covariant_indices = set(i + 1 if i >= axis else i for i in self._covariant_indices)
        result._contravariant_indices = set(i + 1 if i >= axis else i for i in self._contravariant_indices)
        result._collection_indices.add(axis)

        return result

    @property
    def size(self):
        """int: The number of tensors in the collection."""
        return np.prod([self.shape[i] for i in self._collection_indices], dtype=int)

    @property
    def flat(self):
        """generator: A flat iterator of the collection that yields Tensor objects."""
        n_col = len(self._collection_indices)
        covariant = set(i - n_col for i in self._covariant_indices)
        for idx in np.ndindex(self.shape[:n_col]):
            yield self._element_class(self.array[idx], covariant=covariant, copy=False)

    def __getitem__(self, index):
        result = super(TensorCollection, self).__getitem__(index)

        if not isinstance(result, Tensor):
            return result

        if len(result._collection_indices) > 0:
            return TensorCollection(result, copy=False)

        return self._element_class(result, copy=False)

    def __len__(self):
        return len(self.array)


class LeviCivitaTensor(Tensor):
    r"""This class can be used to construct a tensor representing the Levi-Civita symbol.

    The Levi-Civita symbol is also called :math:`\varepsilon`-Tensor and is defined as follows:

    .. math::

        \varepsilon_{\nu_{1} \ldots \nu_{n}} =
        \begin{cases}
            +1 & \text{ if } (\nu_{1}, \ldots, \nu_{n}) \text{ are an even permutation of } (1, \ldots, n)\\
            -1 & \text{ if } (\nu_{1}, \ldots, \nu_{n}) \text{ are an odd permutation of } (1, \ldots, n)\\
            0 & \text{ else}
        \end{cases}

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
        super(LeviCivitaTensor, self).__init__(array, covariant=bool(covariant), copy=False)


class KroneckerDelta(Tensor):
    r"""This class can be used to construct a (p, p)-tensor representing the Kronecker delta tensor.

    The following generalized definition of the Kronecker delta is used:

    .. math::

        \delta_{\nu_{1} \ldots \nu_{p}}^{\mu_{1} \ldots \mu_{p}} =
        \begin{cases}
            +1 & \text{ if } (\nu_{1}, \ldots, \nu_{p}) \text{ are an even permutation of } (\mu_{1}, \ldots, \mu_{p})\\
            -1 & \text{ if } (\nu_{1}, \ldots, \nu_{p}) \text{ are an odd permutation of } (\mu_{1}, \ldots, \mu_{p})\\
            0 & \text{ else}
        \end{cases}

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
            array = np.eye(n, dtype=int)
        elif (p, n) in self._cache:
            array = self._cache[(p, n)]
        elif p == n:
            e = LeviCivitaTensor(n)
            array = np.tensordot(e.array, e.array, 0)

            self._cache[(p, n)] = array
        else:
            d1 = KroneckerDelta(n)
            d2 = KroneckerDelta(n, p - 1)

            def calc(*args):
                return sum(
                    (-1) ** (p + k + 1)
                    * d1.array[args[k], args[-1]]
                    * d2.array[tuple(x for i, x in enumerate(args[:-1]) if i != k)]
                    for k in range(p)
                )

            f = np.vectorize(calc)
            array = np.fromfunction(f, tuple(2 * p * [n]), dtype=int)

        super(KroneckerDelta, self).__init__(array, covariant=range(p), copy=False)


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
        self._index_count += node.rank
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
                    str(source.shape[i]), str(target.shape[j])
                )
            )

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
        result_indices = ([], [], [])
        for i, (node, ind, offset) in enumerate(zip(self._nodes, self._free_indices, self._node_positions)):
            if len(node._collection_indices) > 0:
                col_ind = sorted(node._collection_indices, reverse=True)
                for j, k in enumerate(col_ind[: len(result_indices[0])]):
                    indices[offset + k] = result_indices[0][-j - 1]
                if len(node._collection_indices) > len(result_indices[0]):
                    for k in col_ind[len(result_indices[0]):]:
                        result_indices[0].insert(0, offset + k)
            result_indices[1].extend(offset + x for x in ind[0])
            result_indices[2].extend(offset + x for x in ind[1])
            args.append(node.array)
            s = slice(offset, self._node_positions[i + 1] if i + 1 < len(self._node_positions) else None)
            args.append(indices[s])

        result = np.einsum(*args, result_indices[0] + result_indices[1] + result_indices[2])

        n_col = len(result_indices[0])
        n_cov = len(result_indices[1])

        if n_col > 0:
            return TensorCollection(result, covariant=range(0, n_cov), tensor_rank=result.ndim - n_col, copy=False)

        return Tensor(result, covariant=range(n_cov), copy=False)

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
    """Base class for all projective tensors, i.e. all objects that identify scalar multiples."""

    def __eq__(self, other):
        if np.isscalar(other):
            return super(ProjectiveElement, self).__eq__(other)

        if not isinstance(other, Tensor):
            other = Tensor(other)

        if self.shape != other.shape:
            return False

        return is_multiple(self.array, other.array, rtol=EQ_TOL_REL, atol=EQ_TOL_ABS)

    @property
    def dim(self):
        """int: The ambient dimension of the tensor."""
        return self.shape[0] - 1


class ProjectiveCollection(TensorCollection, ABC):
    """Base class for collections of projective elements."""

    def __eq__(self, other):
        if np.isscalar(other):
            return super(ProjectiveCollection, self).__eq__(other)

        if not isinstance(other, TensorCollection):
            other = TensorCollection(other)

        if self.shape != other.shape:
            return False

        axes = tuple(self._covariant_indices) + tuple(self._contravariant_indices)
        return np.all(is_multiple(self.array, other.array, axis=axes, rtol=EQ_TOL_REL, atol=EQ_TOL_ABS))

    @property
    def dim(self):
        """int: The ambient dimension tensors in the collection."""
        return self.shape[-1] - 1
