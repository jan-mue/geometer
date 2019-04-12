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

    def test_kronecker_delta(self):
        d = KroneckerDelta(3, 4)
        assert d.array.shape == (4,)*6
        assert d.array[0, 1, 2, 0, 1, 2] == 1
        assert d.array[0, 2, 1, 0, 1, 2] == -1
