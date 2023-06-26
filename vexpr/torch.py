import abc

import torch

import vexpr.base as base


class HasDeviceTensors(abc.ABC):
    @abc.abstractmethod
    def to(self, device):
        return NotImplemented()


def move_to_device(device):
    def move(expression):
        if isinstance(expression, HasDeviceTensors):
            expression = expression.to(device)
        return expression
    return move


class SelectFromSymbol(base.Symbol, HasDeviceTensors):
    def __init__(self, name, indices):
        super().__init__(name)
        self.indices = torch.tensor(indices)

    def compute(self, symbols):
        symbol = super().compute(symbols)
        return symbol[self.indices]

    def clone(self):
        return type(self)(self.name, self.indices)

    def to(self, device):
        indices = self.indices.to(device)
        return type(self)(self.name, indices)
