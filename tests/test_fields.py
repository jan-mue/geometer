from geometer.fields import ModularInteger

def test_add():
    a = ModularInteger(7, 31)
    b = ModularInteger(15, 31)
    c = a + b
    assert c == ModularInteger(22, 31)
    assert b + c == ModularInteger(6, 31)

def test_inverse():
    a = ModularInteger(7, 31)
    b = ModularInteger(15, 31)
    assert a.inverse() * a == 1
    assert a/b * b == a
