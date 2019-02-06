from abc import ABC
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

    def __mul__(self, other):
        d = TensorDiagram((other, self))
        return d.calculate()

    def __rmul__(self, other):
        d = TensorDiagram((self, other))
        return d.calculate()

    def __eq__(self, other):
        if other is 0:
            return np.allclose(self.array, np.zeros(self.array.shape))
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
            indices = np.indices(size*[size], dtype=int)
            diff = indices[j] - indices[i]

            # append empty product
            diff = np.append(diff, [np.ones(diff.shape[1:])], axis=0)
            i = np.append(i, [size - 1])

            fact = np.fromiter((np.math.factorial(x) for x in i), dtype=int)
            array = np.prod((diff.T / fact).T, axis=0)

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
        source_index = None
        target_index = None
        index_count = 0

        for ind, tensor in enumerate(self._nodes):

            if tensor is source:
                source_index = index_count
                free_source = self._free_indices[ind]
            if tensor is target:
                target_index = index_count
                free_target = self._free_indices[ind]

            index_count += len(tensor.array.shape)

            if source_index is not None and target_index is not None:
                break

        if source_index is None:
            source_index = index_count
            self._split_indices.append(index_count)
            self._nodes.append(source)
            free_source = {
                "covariant": source._covariant_indices.copy(),
                "contravariant": source._contravariant_indices.copy()
            }
            self._free_indices.append(free_source)
            index_count += len(source.array.shape)

        if target_index is None:
            target_index = index_count
            self._split_indices.append(index_count)
            self._nodes.append(target)
            free_target = {
                "covariant": target._covariant_indices.copy(),
                "contravariant": target._contravariant_indices.copy()
            }
            self._free_indices.append(free_target)
            index_count += len(target.array.shape)

        self._indices.extend(range(len(self._indices), index_count))

        if len(free_source["covariant"]) == 0 or len(free_target["contravariant"]) == 0:
            raise TensorComputationError("Could not add the edge because no free indices are left.")

        i = source_index + free_source["covariant"].pop()
        j = target_index + free_target["contravariant"].pop()
        self._indices[max(i, j)] = min(i, j)

    def calculate(self):
        """Calculates the result of the diagram.

        Returns
        -------
        Tensor
            The tensor resulting from the specified tensor diagram.

        """
        indices = np.split(self._indices, self._split_indices[1:])
        args = [x for i, node in enumerate(self._nodes) for x in (node.array, indices[i].tolist())]
        result_indices = {
            "covariant": [],
            "contravariant": []
        }
        for offset, ind in zip(self._split_indices, self._free_indices):
            result_indices["covariant"].extend(offset + x for x in ind["covariant"])
            result_indices["contravariant"].extend(offset + x for x in ind["contravariant"])
        cov_count = len(result_indices["covariant"])
        result_indices = result_indices["covariant"] + result_indices["contravariant"]
        x = np.einsum(*args, result_indices)
        return Tensor(x, covariant=range(cov_count))


class ProjectiveElement(Tensor, ABC):
    """Base class for all projective tensors, i.e. all objects that identify scalar multiples.

    """

    def __eq__(self, other):
        # By Cauchy-Schwarz |(x,y)| = ||x||*||y|| iff x = cy
        a = self.array.ravel()
        b = other.array.ravel()
        return np.isclose(np.abs(np.vdot(a, b))**2, np.vdot(a, a)*np.vdot(b, b))

    def __len__(self):
        return np.product(self.array.shape)

    @property
    def dim(self):
        """int: The dimension of the tensor."""
        return self.array.shape[0] - 1
