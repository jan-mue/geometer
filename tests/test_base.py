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
