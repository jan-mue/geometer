from __future__ import annotations

from abc import ABC
from collections.abc import Iterable, Iterator, Sequence, Sized
from itertools import permutations
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Generic, Literal

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeVar, Unpack, overload, override

from geometer.exceptions import IncompatibleShapeError, TensorComputationError
from geometer.utils import (
    is_multiple,
    is_numerical_dtype,
    is_numerical_scalar,
    normalize_index,
    posify_index,
    sanitize_index,
)
from geometer.utils.ops_dispatch import maybe_dispatch_ufunc_to_dunder_op

if TYPE_CHECKING:
    from typing_extensions import Self

    from geometer.transformation import Transformation, TransformationCollection, TransformationTensor
    from geometer.utils.typing import NDArrayParameters, NumericalArray, NumericalDType, Shape, TensorIndex

EQ_TOL_REL = 1e-15
EQ_TOL_ABS = 1e-8


class Tensor:
    """Wrapper class around a numpy array that keeps track of covariant and contravariant indices.

    Covariant indices are the lower indices (subscripts) and contravariant indices are the upper indices (superscripts)
    of a tensor.

    Args:
        *args: A single iterable, numpy array, tensor or multiple coordinate numbers, arrays, tensors.
        covariant: If False, all indices are contravariant. If a list of indices is supplied, the specified
            indices of the array will be covariant indices and all others contravariant indices.
            By default, all indices are covariant. If the first argument is a tensor then its indices are copied.
        tensor_rank: If the Tensor object contains multiple tensors, this parameter specifies the rank of the tensors
            contained in it. By default, only a single tensor is contained in a Tensor object.
        **kwargs: Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

    Attributes:
        array: The underlying numpy array.

    References:
      - `Ricci calculus, Upper and lower indices, Wikipedia <https://en.wikipedia.org/wiki/
        Ricci_calculus#Upper_and_lower_indices>`_

    """

    array: NumericalArray
    _covariant_indices: set[int]
    _contravariant_indices: set[int]

    def __init__(
        self,
        *args: Tensor | npt.ArrayLike,
        covariant: bool | Iterable[int] = True,
        tensor_rank: int | None = None,
        **kwargs: Unpack[NDArrayParameters],
    ) -> None:
        if len(args) == 0:
            raise TypeError("At least one argument is required.")

        if len(args) == 1:
            if isinstance(args[0], Tensor):
                self.array = np.array(args[0].array, **kwargs)  # type: ignore[call-overload]
                self._covariant_indices = args[0]._covariant_indices
                self._contravariant_indices = args[0]._contravariant_indices
                return
            else:
                self.array = np.array(args[0], **kwargs)  # type: ignore[call-overload]
        else:
            self.array = np.array(args, **kwargs)  # type: ignore[call-overload]

        if tensor_rank is None:
            tensor_rank = self.rank
        elif tensor_rank < 0:
            tensor_rank += self.shape[-1]

        if self.rank < tensor_rank:
            raise ValueError("tensor_rank must be smaller than or equal to the number of array dimensions.")

        n_free_indices = self.rank - tensor_rank
        if covariant is True:
            self._covariant_indices = set(range(n_free_indices, self.rank))
        elif covariant is False:
            self._covariant_indices = set()
        else:
            self._covariant_indices = set()
            for idx in covariant:
                if not -tensor_rank <= idx < tensor_rank:
                    raise IndexError(f"Index {idx} out of range [{-tensor_rank}, {tensor_rank})")
                idx = sanitize_index(idx)  # type: ignore[no-untyped-call]
                idx = posify_index(tensor_rank, idx)  # type: ignore[no-untyped-call]
                self._covariant_indices.add(n_free_indices + idx)

        free_indices = set(range(n_free_indices))
        self._contravariant_indices = set(range(self.rank)) - self._covariant_indices - free_indices

        self._validate_tensor()

    def _validate_tensor(self) -> None:
        if not is_numerical_dtype(self.dtype):
            raise TypeError(f"The dtype of a Tensor must be a numeric dtype, not {self.dtype.name}")

    @classmethod
    def _get_collection_class(cls) -> type[TensorCollection[Self]]:
        def find_class(current_class: type[TensorCollection[Self]]) -> type[TensorCollection[Self]] | None:
            if current_class._element_class is cls:
                return current_class
            for subclass in current_class.__subclasses__():
                if result_class := find_class(subclass):
                    return result_class
            return None

        result = find_class(TensorCollection)
        if result is None:
            raise TypeError(f"No TensorCollection found for {cls.__name__}")
        return result

    @overload
    def __apply__(self, transformation: Transformation) -> Self: ...

    @overload
    def __apply__(self, transformation: TransformationCollection) -> Self | TensorCollection[Self]: ...

    @overload
    def __apply__(self, transformation: TransformationTensor) -> Self | TensorCollection[Self]: ...

    def __apply__(self, transformation: TransformationTensor) -> Self | TensorCollection[Self]:
        ts = self.tensor_shape
        edges: list[tuple[Tensor, Tensor]] = [(self, transformation.copy()) for _ in range(ts[0])]
        if ts[1] > 0:
            inv = transformation.inverse()
            edges.extend((inv.copy(), self) for _ in range(ts[1]))
        diagram = TensorDiagram(*edges)
        result_tensor = diagram.calculate()
        if result_tensor.free_indices > self.free_indices:
            collection_class = self._get_collection_class()
            return collection_class(result_tensor, copy=None)
        result = self.copy()
        result.array = result_tensor.array
        return result

    @property
    def shape(self) -> Shape:
        """The shape of the underlying numpy array, same as ``self.array.shape``."""
        return self.array.shape

    @property
    def dtype(self) -> NumericalDType:
        """The dtype of the underlying numpy array, same as ``self.array.dtype``."""
        return self.array.dtype

    @property
    def free_indices(self) -> int:
        """Number of free indices, i.e. indices that are not covariant and not contravariant."""
        cov, con = self.tensor_shape
        return self.rank - (cov + con)

    @property
    def tensor_shape(self) -> tuple[int, int]:
        """The shape or type of the tensor.

        The first number is the number of covariant indices, the second the number of contravariant indices.

        """
        return len(self._covariant_indices), len(self._contravariant_indices)

    @property
    def rank(self) -> int:
        """The rank of the Tensor, same as ``self.array.ndim``."""
        return self.array.ndim

    def tensor_product(self, other: Tensor) -> Tensor:
        """Return a new tensor that is the tensor product of this and the other tensor.

        This method will also reorder the indices of the resulting tensor, to ensure that covariant indices are in
        front of the contravariant indices.

        Args:
            other: The other tensor.

        Returns:
            The tensor product.

        """
        if self.free_indices > 0 or other.free_indices > 0:
            raise NotImplementedError("tensor_product is only implemented for tensors without free indices")
        offset = self.rank
        covariant = list(self._covariant_indices) + [offset + i for i in other._covariant_indices]
        contravariant = list(self._contravariant_indices) + [offset + i for i in other._contravariant_indices]

        result = np.tensordot(self.array, other.array, 0)  # type: ignore[arg-type]
        result = np.transpose(result, axes=covariant + contravariant)
        return Tensor(result, covariant=range(len(covariant)), copy=None)

    def transpose(self, perm: Iterable[int] | None = None) -> Tensor:
        """Permute the indices of the tensor. Free indices are not permuted.

        Args:
            perm: A list of permuted indices or a shorter list representing a permutation in cycle notation.
                By default, the indices are reversed.

        Returns:
            The tensor with permuted indices.

        """
        if perm is None:
            perm = list(range(self.free_indices)) + list(reversed(range(self.free_indices, self.rank)))
        else:
            perm = list(perm)

        # TODO: check that free indices are not permuted

        if len(perm) < self.rank:
            a = list(range(self.rank))
            for ind in range(len(perm)):
                i, j = perm[ind], perm[(ind + 1) % len(perm)]
                a[i] = j
            perm = a

        covariant_indices = []
        contravariant_indices = []
        for i, j in enumerate(perm):
            if j in self._covariant_indices:
                covariant_indices.append(i)
            elif j in self._contravariant_indices:
                contravariant_indices.append(i)

        result = Tensor(self.array.transpose(perm), copy=None)
        result._covariant_indices = set(covariant_indices)
        result._contravariant_indices = set(contravariant_indices)

        return result

    @property
    def T(self) -> Tensor:
        """The transposed tensor, same as ``self.transpose()``."""
        return self.transpose()

    def copy(self) -> Self:
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def is_zero(self, tol: float = EQ_TOL_ABS) -> npt.NDArray[np.bool_]:
        """Test whether the tensor is zero with respect to covariant and contravariant indices.

        Args:
            tol: The accepted tolerance.

        Returns:
            True if the tensor is zero. If there are more indices than the covariant and contravariant indices,
            a boolean array is returned.

        """
        axes = tuple(self._covariant_indices) + tuple(self._contravariant_indices)
        return np.all(np.isclose(self.array, 0, atol=tol), axis=axes)

    @override
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.array.tolist()})"

    def _get_index_mapping(self, index: TensorIndex) -> list[int | None]:
        normalized_index = normalize_index(index, self.shape)  # type: ignore[no-untyped-call]
        advanced_indices = []
        index_mapping: list[int | None] = list(range(self.rank))
        i = 0
        for ind in normalized_index:
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

        if len(advanced_indices) == 0:
            return index_mapping

        b = np.broadcast(*[normalized_index[i] for i in advanced_indices])
        a0, a1 = advanced_indices[0], advanced_indices[-1]

        if advanced_indices != list(range(a0, a1 + 1)):
            # create advanced indices in front
            for i in advanced_indices:
                index_mapping.remove(i)
            new_indices: list[int | None] = [None] * b.ndim
            return new_indices + index_mapping
        else:
            # replace indices with broadcast shape
            return index_mapping[:a0] + [None] * b.ndim + index_mapping[a1 + 1 :]

    def __getitem__(self, index: TensorIndex) -> Tensor | np.generic:
        result = self.array[index]

        if isinstance(result, np.generic):
            return result

        covariant_indices = []
        contravariant_indices = []

        index_mapping = self._get_index_mapping(index)

        for new_axis, old_axis in enumerate(index_mapping):
            if old_axis in self._covariant_indices:
                covariant_indices.append(new_axis)
            elif old_axis in self._contravariant_indices:
                contravariant_indices.append(new_axis)

        covariant_indices_set = set(covariant_indices)
        contravariant_indices_set = set(contravariant_indices)

        result_tensor: Self | Tensor
        if (
            self._covariant_indices == covariant_indices_set
            and self._contravariant_indices == contravariant_indices_set
        ):
            result_tensor = self.copy()
            result_tensor.array = result
            result_tensor._validate_tensor()
            return result_tensor

        result_tensor = Tensor(result, copy=None)
        result_tensor._covariant_indices = covariant_indices_set
        result_tensor._contravariant_indices = contravariant_indices_set

        return result_tensor

    def __setitem__(self, key: TensorIndex, value: Tensor | npt.ArrayLike) -> None:
        if isinstance(value, Tensor):
            value = value.array
        self.array[key] = value

    def __copy__(self) -> Self:
        return self.copy()

    def __mul__(self, other: Tensor | npt.ArrayLike) -> Tensor:
        if is_numerical_scalar(other):
            return Tensor(self.array * other, covariant=self._covariant_indices, copy=None)  # type: ignore[operator]
        if not isinstance(other, Tensor):
            other = Tensor(other, copy=None)
        return TensorDiagram((other, self)).calculate()

    def __rmul__(self, other: Tensor | npt.ArrayLike) -> Tensor:
        if is_numerical_scalar(other):
            return self * other  # type: ignore[operator]
        if not isinstance(other, Tensor):
            other = Tensor(other, copy=None)
        return TensorDiagram((self, other)).calculate()

    def __pow__(self, power: int, modulo: int | None = None) -> Tensor:
        if power < 1:
            return NotImplemented
        elif power == 1:
            return self.copy()

        d = TensorDiagram()
        prev = self
        for _ in range(power - 1):
            cur = prev.copy()
            d.add_edge(cur, prev)
            prev = cur

        return d.calculate()

    def __truediv__(self, other: Tensor | npt.ArrayLike) -> Tensor:
        if is_numerical_scalar(other):
            return Tensor(self.array / other, covariant=self._covariant_indices, copy=None)  # type: ignore[operator]
        return NotImplemented

    def __add__(self, other: Tensor | npt.ArrayLike) -> Tensor:
        if isinstance(other, Tensor):
            other = other.array
        return Tensor(self.array + other, covariant=self._covariant_indices, copy=None)  # type: ignore[operator]

    def __radd__(self, other: Tensor | npt.ArrayLike) -> Tensor:
        return self + other

    def __sub__(self, other: Tensor | npt.ArrayLike) -> Tensor:
        if isinstance(other, Tensor):
            other = other.array
        return Tensor(self.array - other, covariant=self._covariant_indices, copy=None)  # type: ignore[operator]

    def __rsub__(self, other: Tensor | npt.ArrayLike) -> Tensor:
        return -self + other

    def __neg__(self) -> Tensor:
        return self * (-1)

    @override
    def __eq__(self, other: object) -> bool:
        other = np.asanyarray(other)
        if self.shape != other.shape:
            return False
        try:
            return np.allclose(self.array, other, rtol=EQ_TOL_REL, atol=EQ_TOL_ABS)
        except TypeError:
            return NotImplemented

    def __array__(self, dtype: npt.DTypeLike | None = None) -> np.ndarray:
        if dtype and dtype != self.dtype:
            return self.array.astype(dtype)
        return self.array

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: Literal["__call__", "reduce", "reduceat", "accumulate", "outer", "inner"],
        *inputs: Any,
        **kwargs: Any,
    ) -> Any:
        return maybe_dispatch_ufunc_to_dunder_op(self, ufunc, method, *inputs, **kwargs)

    def __array_function__(
        self, func: Callable[..., Any], types: Iterable[type], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        return NotImplemented


class BoundTensor(Tensor):
    """A tensor without free indices."""

    @override
    def _validate_tensor(self) -> None:
        super()._validate_tensor()
        if self.free_indices != 0:
            raise IncompatibleShapeError(
                f"Only tensors without free indices can be used in a {self.__class__.__name__}"
            )


TensorT = TypeVar("TensorT", bound=Tensor, covariant=True, default=Tensor)


class TensorCollection(Tensor, Generic[TensorT], Sized, Iterable[TensorT]):
    """A collection of tensors of type TensorT. The shape of axes after axis 0 must be compatible with tensors of type TensorT."""

    _element_class: ClassVar[type[Tensor]] = Tensor

    def __init__(
        self,
        *args: Tensor | npt.ArrayLike,
        covariant: bool | Iterable[int] = True,
        tensor_rank: int | None = 1,
        **kwargs: Unpack[NDArrayParameters],
    ) -> None:
        super().__init__(*args, covariant=covariant, tensor_rank=tensor_rank, **kwargs)

    @override
    def _validate_tensor(self) -> None:
        super()._validate_tensor()
        if self.free_indices == 0:
            raise IncompatibleShapeError(
                f"Tensor has no free indices and cannot be used in a {self.__class__.__name__}."
            )

    @classmethod
    def from_tensor(cls, tensor: Tensor, **kwargs: Unpack[NDArrayParameters]) -> Self | TensorT:
        """Construct an object from another tensor. If the tensor has no free indices, an object of type TensorT is returned.

        By default the array in the tensor is not copied.

        Args:
            tensor: A tensor to use for the new tensor.
            **kwargs: Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

        Returns:
            A new tensor.

        """
        kwargs.setdefault("copy", False)
        if tensor.free_indices > 0:
            return cls(tensor, **kwargs)
        else:
            return cls._element_class(tensor, **kwargs)  # type: ignore[return-value]

    @classmethod
    def from_array(cls, array: npt.ArrayLike, **kwargs: Unpack[NDArrayParameters]) -> Self | TensorT:
        """Try to construct a new collection from an array. If the rank is too low, an object of type TensorT is returned.

        By default the array is not copied.

        Args:
            array: A numpy array to use for the new tensor.
            **kwargs: Additional keyword arguments for the constructor of the numpy array as defined in `numpy.array`.

        Returns:
            A new tensor.

        """
        kwargs.setdefault("copy", False)
        try:
            return cls(array, **kwargs)
        except IncompatibleShapeError:
            return cls._element_class(array, **kwargs)  # type: ignore[return-value]

    def expand_dims(self, axis: int) -> Self:
        """Add a new index as a free index.

        Args:
            axis: Position in the new shape where the new axis is placed.

        Returns:
            The tensor collection with an additional free index.

        """
        axis = sanitize_index(axis)  # type: ignore[no-untyped-call]
        axis = posify_index(self.rank + 1, axis)  # type: ignore[no-untyped-call]
        if axis > self.free_indices:
            raise ValueError("Free indices can only be inserted at the beginning.")

        result = self.copy()
        result.array = np.expand_dims(self.array, axis)
        result._covariant_indices = {i + 1 if i >= axis else i for i in self._covariant_indices}
        result._contravariant_indices = {i + 1 if i >= axis else i for i in self._contravariant_indices}
        return result

    @property
    def size(self) -> npt.NDArray[np.int_]:
        """The number of tensors in the tensor collection, i.e. the product of the size of all collection axes."""
        return np.prod([self.shape[i] for i in range(self.free_indices)], dtype=int)

    @overload
    def __getitem__(self, index: int | np.int_) -> TensorT: ...

    @overload
    def __getitem__(self, index: Sequence[int] | Sequence[np.int_] | Sequence[bool] | Sequence[np.bool_]) -> Self: ...

    @overload
    def __getitem__(self, index: npt.NDArray[np.int_] | npt.NDArray[np.bool_]) -> Tensor: ...

    @overload
    def __getitem__(self, index: TensorIndex) -> Tensor | np.generic: ...

    @override
    def __getitem__(self, index: TensorIndex) -> Tensor | np.generic:
        result = super().__getitem__(index)

        if not isinstance(result, Tensor):
            return result

        if isinstance(index, (int, np.int_)):
            return self._element_class(result, copy=None)

        if result.free_indices == 0 or isinstance(result, type(self)):
            return result

        return TensorCollection(result, copy=None)

    @override
    def __len__(self) -> int:
        return len(self.array)

    @override
    def __iter__(self) -> Iterator[TensorT]:
        for i in range(len(self)):
            yield self[i]


class LeviCivitaTensor(BoundTensor):
    r"""This class can be used to construct a tensor representing the Levi-Civita symbol.

    The Levi-Civita symbol is also called :math:`\varepsilon`-Tensor and is defined as follows:

    .. math::

        \varepsilon_{\nu_{1} \ldots \nu_{n}} =
        \begin{cases}
            +1 & \text{ if } (\nu_{1}, \ldots, \nu_{n}) \text{ are an even permutation of } (1, \ldots, n)\\
            -1 & \text{ if } (\nu_{1}, \ldots, \nu_{n}) \text{ are an odd permutation of } (1, \ldots, n)\\
            0 & \text{ else}
        \end{cases}

    Args:
        size: The number of indices of the tensor.
        covariant: If true, the tensor will only have covariant indices. Default: True

    References:
      - `Levi-Civita symbol, Wikipedia <https://en.wikipedia.org/wiki/
        Levi-Civita_symbol#Generalization_to_n_dimensions>`_

    """

    array: npt.NDArray[np.int8]
    _cache: ClassVar[dict[int, np.ndarray]] = {}

    def __init__(self, size: int, covariant: bool = True) -> None:
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
        super().__init__(array, covariant=bool(covariant), copy=None)


class KroneckerDelta(BoundTensor):
    r"""This class can be used to construct a (p, p)-tensor representing the Kronecker delta tensor.

    The following generalized definition of the Kronecker delta is used:

    .. math::

        \delta_{\nu_{1} \ldots \nu_{p}}^{\mu_{1} \ldots \mu_{p}} =
        \begin{cases}
            +1 & \text{ if } (\nu_{1}, \ldots, \nu_{p}) \text{ are an even permutation of } (\mu_{1}, \ldots, \mu_{p})\\
            -1 & \text{ if } (\nu_{1}, \ldots, \nu_{p}) \text{ are an odd permutation of } (\mu_{1}, \ldots, \mu_{p})\\
            0 & \text{ else}
        \end{cases}

    Args:
        n: The dimension of the tensor.
        p: The number of covariant and contravariant indices of the tensor, default is 1.

    References:
      - `Kronecker delta, Wikipedia <https://en.wikipedia.org/wiki/Kronecker_delta#Generalizations>`_

    """

    array: npt.NDArray[np.int_]
    _cache: ClassVar[dict[tuple[int, int], np.ndarray]] = {}

    def __init__(self, n: int, p: int = 1) -> None:
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

            def calc(*args: int) -> int:
                return sum(
                    (-1) ** (p + k + 1)
                    * d1.array[args[k], args[-1]]
                    * d2.array[tuple(x for i, x in enumerate(args[:-1]) if i != k)]
                    for k in range(p)
                )

            f = np.vectorize(calc)
            array = np.fromfunction(f, tuple(2 * p * [n]), dtype=int)

        super().__init__(array, covariant=range(p), copy=None)


class TensorDiagram:
    """A class used to specify and calculate tensor diagrams (also called Penrose Graphical Notation).

    Each edge in the diagram represents a contraction of two indices of the tensors connected by that edge. In
    Einstein-notation that would mean that an edge from tensor A to tensor B is equivalent to the expression
    :math:`A_{i j}B^{i k}_l`, where :math:`j, k, l` are free indices. The indices to contract are chosen from front to
    back from contravariant and covariant indices of the tensors that are connected by an edge.

    Args:
        *edges: Variable number of tuples, that represent the edge from one tensor to another.


    References:
      - https://www-m10.ma.tum.de/foswiki/pub/Lehrstuhl/PublikationenJRG/52_TensorDiagrams.pdf
      - J. Richter-Gebert: Perspectives on Projective Geometry, Chapters 13-14

    """

    def __init__(self, *edges: tuple[Tensor, Tensor]) -> None:
        self._nodes: list[Tensor] = []
        self._unused_indices: list[tuple[list[int], list[int]]] = []
        self._node_positions: list[int] = []
        self._contraction_list: list[tuple[int, int, int, int]] = []
        self._index_count: int = 0

        for e in edges:
            self.add_edge(*e)

    def add_node(self, node: Tensor) -> tuple[list[int], list[int]]:
        """Add a node to the tensor diagram without adding an edge/contraction.

        A diagram of nodes where none are connected is equivalent to calculating the tensor product with the
        method :meth:`Tensor.tensor_product`.

        Args:
            node: The node to add.

        """
        self._nodes.append(node)
        self._node_positions.append(self._index_count)
        self._index_count += node.rank
        ind = (list(node._covariant_indices), list(node._contravariant_indices))
        self._unused_indices.append(ind)
        return ind

    def add_edge(self, source: Tensor, target: Tensor) -> None:
        """Add an edge to the diagram.

        Args:
            source: The source tensor of the edge in the diagram.
            target: The target tensor of the edge in the diagram.

        Raises:
            TensorComputationError: If the tensors have no unused indices or the dimensions do not match.

        """
        # First step: Find nodes if they are already in the diagram
        source_index, target_index = None, None
        for index, node in enumerate(self._nodes):
            if node is source:
                source_index = index
                free_source = self._unused_indices[index][0]
            if node is target:
                target_index = index
                free_target = self._unused_indices[index][1]

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
            raise TensorComputationError("Could not add the edge because no indices are left.")

        # Third step: Pick some free indices
        i = free_source.pop(0)
        j = free_target.pop(0)

        if source.shape[i] != target.shape[j]:
            raise TensorComputationError(
                f"Dimension of tensors is inconsistent, encountered dimensions {source.shape[i]} and {target.shape[j]}."
            )

        self._contraction_list.append((source_index, target_index, i, j))

    def calculate(self) -> BoundTensor | TensorCollection:
        """Calculates the result of the diagram.

        Returns:
            The tensor resulting from the specified tensor diagram.

        """
        # Build the list of indices for einsum
        indices = list(range(self._index_count))
        for source_index, target_index, i, j in self._contraction_list:
            i = self._node_positions[source_index] + i
            j = self._node_positions[target_index] + j
            indices[max(i, j)] = min(i, j)

        # Split the indices and insert the arrays
        args: list[npt.ArrayLike] = []
        result_indices: tuple[list[int], list[int], list[int]] = ([], [], [])
        for i, (node, ind, offset) in enumerate(zip(self._nodes, self._unused_indices, self._node_positions)):
            if node.free_indices > 0:
                free_ind = list(reversed(range(node.free_indices)))
                for j, k in enumerate(free_ind[: len(result_indices[0])]):
                    indices[offset + k] = result_indices[0][-j - 1]
                if node.free_indices > len(result_indices[0]):
                    for k in free_ind[len(result_indices[0]) :]:
                        result_indices[0].insert(0, offset + k)
            result_indices[1].extend(offset + x for x in ind[0])
            result_indices[2].extend(offset + x for x in ind[1])
            args.append(node.array)
            s = slice(offset, self._node_positions[i + 1] if i + 1 < len(self._node_positions) else None)
            args.append(indices[s])

        result = np.einsum(*args, result_indices[0] + result_indices[1] + result_indices[2])  # type: ignore[arg-type]

        n_free = len(result_indices[0])
        n_cov = len(result_indices[1])

        if n_free > 0:
            return TensorCollection(result, covariant=range(n_cov), tensor_rank=result.ndim - n_free, copy=False)

        return BoundTensor(result, covariant=range(n_cov), copy=None)

    def copy(self) -> TensorDiagram:
        result = TensorDiagram()
        result._nodes = self._nodes.copy()
        result._unused_indices = self._unused_indices.copy()
        result._node_positions = self._node_positions.copy()
        result._contraction_list = self._contraction_list.copy()
        result._index_count = self._index_count
        return result

    def __copy__(self) -> TensorDiagram:
        return self.copy()


class ProjectiveTensor(Tensor, ABC):
    """Base class for all projective tensors, i.e. all objects that identify scalar multiples."""

    def __init__(
        self,
        *args: Tensor | npt.ArrayLike,
        covariant: bool | Iterable[int] = True,
        tensor_rank: int | None = None,
        **kwargs: Unpack[NDArrayParameters],
    ) -> None:
        super().__init__(*args, covariant=covariant, tensor_rank=tensor_rank, **kwargs)
        if self.rank == 0:
            raise ValueError("A projective tensor cannot be of rank 0")
        if self.shape[-1] == 0:
            raise ValueError("A projective tensor cannot have trailing dimension 0")
        if not 0 < self.dim <= 3:
            raise ValueError(f"Only dimensions 1, 2 and 3 are supported, got dimension {self.dim}")

    @override
    def __eq__(self, other: object) -> bool:
        try:
            is_scalar = is_numerical_scalar(other)  # type: ignore[arg-type]
        except TypeError:
            # other is not an array-like
            return NotImplemented

        if is_scalar:
            return super().__eq__(other)

        if not isinstance(other, Tensor):
            other = Tensor(other)  # type: ignore[arg-type]

        if self.shape != other.shape:
            return False

        axes = tuple(self._covariant_indices) + tuple(self._contravariant_indices)
        try:
            is_multi = is_multiple(self.array, other.array, axis=axes, rtol=EQ_TOL_REL, atol=EQ_TOL_ABS)
            return bool(np.all(is_multi))
        except TypeError:
            return NotImplemented

    @property
    def dim(self) -> int:
        """The ambient dimension of the tensor."""
        return self.shape[-1] - 1
