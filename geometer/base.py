from abc import ABC, abstractmethod
import numpy as np


class Tensor:

    def __init__(self, *args, contravariant_indices=None):
        if len(args) == 1:
            if isinstance(args[0], Tensor):
                self.array = args[0].array
                contravariant_indices = args[0]._contravariant_indices
            else:
                self.array = np.atleast_1d(args[0])
        else:
            self.array = np.array([*args])

        self._contravariant_indices = set(contravariant_indices or [])
        self._covariant_indices = set(range(len(self.array.shape))) - self._contravariant_indices

    def __mul__(self, other):
        d = TensorDiagram((other, self))
        return d.calculate()

    def __rmul__(self, other):
        d = TensorDiagram((self, other))
        return d.calculate()

    def __eq__(self, other):
        if other is 0:
            return np.vectorize(lambda x: np.isclose(x, 0, atol=0.0))(self.array).all()
        return self.array == other


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
        self._indices = []
        self._free_indices = []
        self._split_indices = []
        for e in edges or []:
            self.add_edge(*e)

    def add_edge(self, source, target):
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
            raise ValueError("Could not add edge.")

        i = source_index + free_source["covariant"].pop()
        j = target_index + free_target["contravariant"].pop()
        self._indices[max(i, j)] = min(i, j)

    def calculate(self):
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
        x = np.einsum(*args, result_indices, optimize="optimal")
        return Tensor(x, contravariant_indices=range(cov_count, len(result_indices)))


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

    @property
    def dim(self):
        return self.array.shape[0] - 1
