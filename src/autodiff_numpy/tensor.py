import numpy as np


class Tensor:
    """
    A class to represent a Tensor, which is a node in the computation graph.
    It holds data, gradient, and tracks the operation that created it.
    """

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = np.asarray(data, dtype=np.float32)
        self.gradient = np.zeros_like(self.data)
        self._backward = lambda: None
        self._previous_nodes = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Tensor(data={self.data.shape}, grad={self.gradient.shape})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            # self.grad
            grad_for_self = out.gradient
            if self.gradient.shape != grad_for_self.shape:
                # sum of gradient if broadcasting occurred
                self.gradient += np.sum(grad_for_self, axis=0, keepdims=self.gradient.ndim > 1)
            else:
                self.gradient += grad_for_self

            # other.grad
            grad_for_other = out.gradient
            if other.gradient.shape != grad_for_other.shape:
                # sum gradient if broadcasting occurred
                other.gradient += np.sum(grad_for_other, axis=0, keepdims=other.gradient.ndim > 1)
            else:
                other.gradient += grad_for_other

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            # self.grad
            grad_for_self = other.data * out.gradient
            if self.gradient.shape != grad_for_self.shape:
                # sum the incoming gradient to match the operand's shape
                self.gradient += np.sum(grad_for_self)
            else:
                self.gradient += grad_for_self

            # other.grad
            grad_for_other = self.data * out.gradient
            if other.gradient.shape != grad_for_other.shape:
                # sum the incoming gradient to match the operand's shape
                other.gradient += np.sum(grad_for_other)
            else:
                other.gradient += grad_for_other

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only supporting power with scalar for now"
        out = Tensor(self.data**other, (self,), f"**{other}")

        def _backward():
            self.gradient += (other * self.data ** (other - 1)) * out.gradient

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            self.gradient += out.gradient @ other.data.T
            other.gradient += self.data.T @ out.gradient

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), "ReLU")

        def _backward():
            self.gradient += (out.data > 0) * out.gradient

        out._backward = _backward
        return out

    def sigmoid(self):
        out = Tensor(1 / (1 + np.exp(-self.data)), (self,), "sigmoid")

        def _backward():
            self.gradient *= out.data * (1 - out.data)

        out._backward = _backward
        return out

    def backward(self):
        # recursive topological sort of the graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._previous_nodes:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # apply the chain rule to get its gradient (from the last node of the graph to the first)
        self.gradient = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def sum(self):
        out = Tensor(np.sum(self.data), (self,), "sum")

        def _backward():
            self.gradient += np.ones_like(self.data) * out.gradient

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        return self + (-1 * other)

    def __rsub__(self, other):
        return other + (-1 * self)
