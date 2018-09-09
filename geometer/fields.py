from abc import ABC, abstractmethod
from .utils import xgcd


class FieldElement(ABC):

    @abstractmethod
    def __add__(self, other): pass

    @abstractmethod
    def __mul__(self, other): pass

    @abstractmethod
    def __neg__(self): pass

    @abstractmethod
    def inverse(self): pass

    def __sub__(self, other): return self + (-other)

    def __radd__(self, other): return self + other

    def __rsub__(self, other): return -self + other

    def __rmul__(self, other): return self * other

    def __truediv__(self, other): return self * other.inverse()

    def __rtruediv__(self, other): return self.inverse() * other

    def __div__(self, other): return self.__truediv__(other)

    def __rdiv__(self, other): return self.__rtruediv__(other)


class ModularInteger(FieldElement):

    def __init__(self, number, p):
        self.number = int(number) % p
        self.p = p

    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = ModularInteger(other, self.p)
        if not isinstance(other, ModularInteger):
            return NotImplemented
        return ModularInteger(self.number + other.number, self.p)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = ModularInteger(other, self.p)
        if not isinstance(other, ModularInteger):
            return NotImplemented
        return ModularInteger(self.number - other.number, self.p)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = ModularInteger(other, self.p)
        if not isinstance(other, ModularInteger):
            return NotImplemented
        return ModularInteger(self.number * other.number, self.p)

    def __pow__(self, power):
        return ModularInteger(pow(self.number, power, self.p), self.p)

    def __neg__(self): return ModularInteger(-self.number, self.p)

    def __eq__(self, other):
        if isinstance(other, (int, float)):
            return self.number == other
        return isinstance(other, ModularInteger) and self.number == other.number

    def __abs__(self): return abs(self.number)

    def __str__(self): return str(self.number)

    def __repr__(self): return '%d (mod %d)' % (self.number, self.p)

    def __divmod__(self, divisor):
        q, r = divmod(self.number, divisor.number)
        return ModularInteger(q, self.p), ModularInteger(r, self.p)

    def __truediv__(self, other):
        other = ModularInteger(other, self.p)
        return self * other.inverse()

    def __int__(self): return int(self.number)

    def __float__(self): return float(self.number)

    def inverse(self):
        d,x,y = xgcd(self.number, self.p)
        return ModularInteger(x, self.p)

