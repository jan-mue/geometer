from geometer.base import TensorDiagram, Tensor, KroneckerDelta


class TestTensorDiagram:

    def test_add_edge(self):
        a = Tensor([1, 0, 0, 0])
        b = Tensor([[42, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]], covariant=False)
        diagram = TensorDiagram((a, b))
        assert diagram.calculate() == Tensor([42, 0, 0, 0])
        diagram.add_edge(a.copy(), b)
        assert diagram.calculate() == 42

    def test_tensor_product(self):
        e1 = Tensor(1, 0)
        e2 = Tensor(0, 1)
        a = Tensor([0, 1],
                   [1, 0], covariant=[0])
        b = Tensor([1, 0],
                   [0, 1], covariant=[0])

        m = a.tensor_product(b)
        e = e1.tensor_product(e2)
        assert TensorDiagram((e, m), (e, m)).calculate() == (a * e1).tensor_product(b * e2)

    def test_kronecker_delta(self):
        d = KroneckerDelta(4, 3)
        assert d.array.shape == (4,)*6
        assert d.array[0, 1, 2, 0, 1, 2] == 1
        assert d.array[0, 2, 1, 0, 1, 2] == -1
