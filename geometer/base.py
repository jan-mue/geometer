from abc import ABC, abstractmethod
import numpy as np


class Tensor:

    def __init__(self, *args, contravariant_indices=None):
        if len(args) == 1:
            self.array = np.atleast_1d(args[0])
        else:
            self.array = np.array([*args])

        self._contravariant_indices = set(contravariant_indices or [])
        self._covariant_indices = set(range(len(self.array.shape))) - self._contravariant_indices


class LeviCivitaTensor(Tensor):
    
    def __init__(self, size, covariant=True):
        contravariant_indices = range(size) if not covariant else None
        f = np.vectorize(self._calc)
        array = np.fromfunction(f, tuple(size*[size]), dtype=int)
        super(LeviCivitaTensor, self).__init__(array, contravariant_indices=contravariant_indices)

    @staticmethod
    def _calc(*args):
        n = len(args)
        return np.prod([np.prod([args[j] - args[i] for j in range(i + 1, n)]) / np.math.factorial(i) for i in range(n)])


class TensorDiagram:

    def __init__(self, *edges):
        self._nodes = []
        self._adjacency_matrix = np.zeros((2,2), dtype=int)
        for e in edges or []:
            self.add_edge(*e)

    def add_edge(self, source, target):
        source_index = None
        target_index = None
        index_count = 0

        for tensor in self._nodes:

            if tensor is source:
                source_index = index_count
            if tensor is target:
                target_index = index_count

            index_count += len(tensor.array.shape)

            if source_index is not None and target_index is not None:
                break

        if source_index is None:
            source_index = index_count
            self._nodes.append(source)
            index_count += len(source.array.shape)

        if target_index is None:
            target_index = index_count
            self._nodes.append(target)
            index_count += len(target.array.shape)

        n = index_count - self._adjacency_matrix.shape[0]
        self._adjacency_matrix = np.pad(self._adjacency_matrix, ((0, n), (0, n)), "constant")

        for i in source._covariant_indices:
            if not self._adjacency_matrix[source_index+i, :].any():
                break
        else:
            raise ValueError("Could not add edge.")

        for j in target._contravariant_indices:
            if not self._adjacency_matrix[:, target_index+j].any():
                break
        else:
            raise ValueError("Could not add edge.")

        self._adjacency_matrix[source_index + i, target_index + j] = 1

    def calculate(self):
        indices = np.arange(1, self._adjacency_matrix.shape[0]+1)
        x = self._adjacency_matrix.dot(indices)

        for i, j in np.ndenumerate(x):
            if j != 0:
                indices[i] = j

        i = 0
        s = []
        for n in self._nodes[:-1]:
            i += len(n.array.shape)
            s.append(i)

        return np.einsum(*[x for i, node in enumerate(self._nodes) for x in (node.array, np.split(indices, s)[i].tolist())], optimize="optimal")


class GeometryObject(ABC):

    @abstractmethod
    def intersect(self, other):
        pass


class ProjectiveElement(GeometryObject, Tensor, ABC):

    def __eq__(self, other):
        # By Cauchy-Schwarz |(x,y)| = ||x||*||y|| iff x = cy
        a = self.array.ravel()
        b = other.array.ravel()
        return np.isclose(float(np.abs(np.vdot(a, b)))**2, float(np.vdot(a, a)*np.vdot(b, b)))

    def __len__(self):
        return np.product(self.array.shape)
