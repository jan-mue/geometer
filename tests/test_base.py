from geometer.base import TensorDiagram, Tensor, LeviCivitaTensor, KroneckerDelta


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
        assert a / 6 == Tensor(1/3, 0.5)


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
        assert d.array.shape == (4,)*6
        assert d.array[0, 1, 2, 0, 1, 2] == 1
        assert d.array[0, 2, 1, 0, 1, 2] == -1
