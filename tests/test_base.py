from geometer.base import *


class TestTensorDiagram:

    def test_add_edge(self):
        a = Tensor([1, 0, 0, 0])
        b = Tensor([[42, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]], covariant=False)
        c = TensorDiagram((a, b))
        assert c == Tensor([42, 0, 0, 0])
        c.add_edge(a.copy(), b)
        assert c == 42

    def test_kronecker_delta(self):
        d = KroneckerDelta(3, 4)
        assert d.array.shape == (4,)*6
        assert d.array[0, 1, 2, 0, 1, 2] == 1
        assert d.array[0, 2, 1, 0, 1, 2] == -1
