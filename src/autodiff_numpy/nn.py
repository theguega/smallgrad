import numpy as np

from autodiff_numpy.tensor import Tensor


class Module:
    """Base class for all neural network modules."""

    def zero_grad(self):
        for p in self.parameters():
            p.gradient = np.zeros_like(p.data)

    def parameters(self):
        return []


class Linear(Module):
    """A fully-connected linear layer."""

    def __init__(self, in_features, out_features):
        self.weight = Tensor(
            np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        )
        self.bias = Tensor(np.zeros((1, out_features)))

    def __call__(self, x):
        self.out = x @ self.weight + self.bias
        return self.out

    def parameters(self):
        return [self.weight, self.bias]


class ReLU(Module):
    """ReLU activation function."""

    def __call__(self, x):
        return x.relu()

    def parameters(self):
        return []


class Sigmoid(Module):
    """Sigmoid activation function."""

    def __call__(self, x):
        return x.sigmoid()

    def parameters(self):
        return []


class Sequential(Module):
    """A sequential container for modules."""

    def __init__(self, *layers):
        self.layers = list(layers)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.gradient.fill(0)
