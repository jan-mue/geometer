import numpy as np
from sympy.abc import a, b, c, d, e, f, g, h, i
from geometer.base import TensorDiagram, Tensor


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

    def test_sympy_symbols(self):
        arr = np.array([[a, b, c],
                        [d, e, f],
                        [g, h, i]])

        diagram = TensorDiagram((Tensor(arr, covariant=[0]), Tensor(arr, covariant=[0])))

        assert diagram == Tensor(arr.dot(arr))
