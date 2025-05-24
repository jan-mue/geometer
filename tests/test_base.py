import numpy as np

from geometer.base import KroneckerDelta, LeviCivitaTensor, Tensor, TensorCollection, TensorDiagram


class TestTensor:
    def test_arithmetic(self):
        a = Tensor(2, 3)
        b = Tensor(5, 4)

        # vector operations
        assert a + b == Tensor(7, 7)
        assert a - b == Tensor(-3, -1)
        assert -a == Tensor(-2, -3)

        # scalar operations
        assert a + 6 == Tensor(8, 9)
        assert a - 6 == Tensor(-4, -3)
        assert a * 6 == Tensor(12, 18)
        assert a / 6 == Tensor(1 / 3, 0.5)

    def test_transpose(self):
        a = Tensor([[1, 2], [3, 4]], covariant=[0])

        assert a.transpose() == Tensor([[1, 3], [2, 4]])
        assert a.T._covariant_indices == {1}
        assert a == a.T.T

    def test_getitem(self):
        a = Tensor([[1, 2], [3, 4]], covariant=[0])

        assert a[0, 1] == 2
        assert a[None, 1] == [[3, 4]]
        assert a[None, 1].tensor_shape == (0, 1)
        assert a[::-1, 0] == [3, 1]
        assert a[::-1, 0].tensor_shape == (1, 0)

    def test_dtype(self):
        a = Tensor(2, 3, dtype=np.float32)
        assert a.dtype == np.float32

        a = Tensor(2, 3, dtype=np.complex64)
        assert a.dtype == np.complex64


class TestTensorCollection:
    def test_init(self):
        # numpy array
        a = TensorCollection[Tensor](np.ones((1, 2, 3)), tensor_rank=1)
        assert len(a) == 1
        assert a.size == 2

        # nested list of numbers
        a = TensorCollection[Tensor]([[1, 2], [3, 4]], tensor_rank=1)
        assert len(a) == 2
        assert a.size == 2

        # nested tuple of numbers
        a = TensorCollection[Tensor](((1, 2), (3, 4)), tensor_rank=1)
        assert len(a) == 2
        assert a.size == 2

        # nested list of Tensor objects
        a = TensorCollection[Tensor]([[Tensor(1, 2, 3), Tensor(3, 4, 5)]], tensor_rank=1)
        assert a.shape == (1, 2, 3)
        assert len(a) == 1
        assert a.size == 2

        # object with __array__ function
        class A:
            def __array__(self):
                return np.array([Tensor(1, 2), Tensor(3, 4)])

        a = TensorCollection[Tensor](A(), tensor_rank=1)
        assert len(a) == 2
        assert a.size == 2

    def test_getitem(self):
        a = TensorCollection[Tensor]([[1, 2], [3, 4]])
        b = TensorCollection[Tensor]([[5, 6], [7, 8]])
        c = TensorCollection[Tensor]([a, b], tensor_rank=1)

        assert c[0] == a
        assert c[1] == b
        assert list(c) == [a, b]
        assert c[:, 1] == Tensor([Tensor([3, 4]), Tensor([7, 8])], tensor_rank=1)
        assert c[:, 0, 0] == [1, 5]

    def test_expand_dims(self):
        a = TensorCollection[Tensor]([[1, 2], [3, 4]])
        b = a.expand_dims(0)

        assert b.shape == (1, 2, 2)
        assert b[0] == a

        c = a.expand_dims(-2)

        assert c.shape == (2, 1, 2)
        assert c[:, 0, :] == a

        d = a.expand_dims(-3)

        assert d.shape == (1, 2, 2)
        assert d[0] == a


class TestTensorDiagram:
    def test_add_edge(self):
        a = Tensor([1, 0, 0, 0])
        b = Tensor(
            [
                [42, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            covariant=False,
        )
        diagram = TensorDiagram((a, b))
        assert diagram.calculate() == Tensor([42, 0, 0, 0])
        diagram.add_edge(a.copy(), b)
        assert diagram.calculate() == 42

    def test_tensor_product(self):
        e1 = Tensor(1, 0)
        e2 = Tensor(0, 1)
        a = Tensor([0, 1], [1, 0], covariant=[0])
        b = Tensor([1, 0], [0, 1], covariant=[0])

        m = a.tensor_product(b)
        e = e1.tensor_product(e2)
        assert TensorDiagram((e, m), (e, m)).calculate() == (a * e1).tensor_product(b * e2)

        d = TensorDiagram()
        d.add_node(a)
        d.add_node(b)
        assert d.calculate() == a.tensor_product(b)

    def test_epsilon_delta_rule(self):
        e1 = LeviCivitaTensor(3, True)
        e2 = LeviCivitaTensor(3, False)
        d = KroneckerDelta(3)
        d2 = d.tensor_product(d)
        d1 = d2.transpose((0, 1))

        diagram = TensorDiagram((e1, e2.transpose()))
        assert diagram.calculate() == d1 - d2

    def test_kronecker_delta(self):
        d = KroneckerDelta(4, 3)
        assert d.array.shape == (4,) * 6
        assert d.array[0, 1, 2, 0, 1, 2] == 1
        assert d.array[0, 2, 1, 0, 1, 2] == -1
