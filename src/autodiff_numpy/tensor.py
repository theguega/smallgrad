import numpy as np


class Tensor:
    """
    A class to represent a Tensor, which is a node in the computation graph.
    It holds data, gradient, and tracks the operation that created it.
    """

    def __init__(self, data, _children=(), _op=""):
        self.data = np.asarray(data)
        self.gradient = np.zeros_like(self.data)
        self._backward = lambda: None
        self._previous_nodes = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Tensor(data={self.data.shape}, grad={self.gradient.shape}, op={self._op})"

    def __add__(self, b):
        """
        self : tensor a
        input : tensor b
        return : c = a + b
        """
        b = b if isinstance(b, Tensor) else Tensor(b)
        c = Tensor(self.data + b.data, (self, b))

        def _backward():
            """
            a_grad = c_grad * 1
            b_grad = c_grad * 1
            """
            a_grad = c.gradient
            b_grad = c.gradient

            if self.gradient.shape != a_grad.shape:
                keepdims = len(self.gradient.shape) == len(a_grad.shape)
                self.gradient += np.sum(a_grad, axis=0, keepdims=keepdims)
            else:
                self.gradient = self.gradient + a_grad

            if b.gradient.shape != b_grad.shape:
                keepdims = len(b.gradient.shape) == len(b_grad.shape)
                b.gradient = b.gradient + np.sum(b_grad, axis=0, keepdims=keepdims)
            else:
                b.gradient = b.gradient + b_grad

        c._backward = _backward
        return c

    def __mul__(self, b):
        """
        self : tensor a
        input : tensor b
        return : c = a * b
        """
        b = b if isinstance(b, Tensor) else Tensor(b)
        c = Tensor(self.data * b.data, (self, b))

        def _backward():
            """
            a_grad = c_grad * b
            b_grad = c_grad * a
            """
            a_grad = b.data * c.gradient
            b_grad = self.data * c.gradient

            if self.gradient.shape != a_grad.shape:
                keepdims = len(self.gradient.shape) == len(a_grad.shape)
                self.gradient = self.gradient + np.sum(a_grad, axis=0, keepdims=keepdims)
            else:
                self.gradient += a_grad

            if b.gradient.shape != b_grad.shape:
                keepdims = len(b.gradient.shape) == len(b_grad.shape)
                b.gradient = b.gradient + np.sum(b_grad, axis=0, keepdims=keepdims)
            else:
                b.gradient = b.gradient + b_grad

        c._backward = _backward
        return c

    def __pow__(self, n):
        """
        self : tensor a
        input : int n (only integer and float are supported)
        return : c = a ^ n
        """
        assert isinstance(n, (int, float))
        c = Tensor(self.data**n, (self,))

        def _backward():
            """
            a_grad = c_grad * n * (a ** (n - 1))
            """
            self.gradient = self.gradient + (n * self.data ** (n - 1)) * c.gradient

        c._backward = _backward
        return c

    def __matmul__(self, b):
        """
        self : tensor a
        input : tensor b
        return : c = a @ b
        """
        b = b if isinstance(b, Tensor) else Tensor(b)
        c = Tensor(self.data @ b.data, (self, b))

        def _backward():
            """
            a_grad = c_grad @ b.T
            b_grad = a.T @ c_grad
            """
            self.gradient = self.gradient + c.gradient @ b.data.T
            b.gradient = b.gradient + self.data.T @ c.gradient

        c._backward = _backward
        return c

    def relu(self):
        """
        self : tensor a
        return : c = a if a > 0 else 0
        """
        c = Tensor(np.maximum(0, self.data), (self,))

        def _backward():
            """
            a_grad = c_grad * 1 if a > 0 else 0
            """
            a_grad = c.gradient * (self.data > 0)
            self.gradient = self.gradient + a_grad

        c._backward = _backward
        return c

    def sigmoid(self):
        """
        input : tensor a
        return : c = 1 / (1 + exp(-a))
        """
        c = Tensor(1 / (1 + np.exp(-self.data)), (self,))

        def _backward():
            """
            a_grad = c_grad * c * (1 - c)
            """
            a_grad = c.gradient * c.data * (1 - c.data)
            self.gradient = self.gradient + a_grad

        c._backward = _backward
        return c

    def sum(self):
        """
        input : tensor a
        return : c = sum(a)
        """
        c = Tensor(np.sum(self.data), (self,))

        def _backward():
            """
            a_grad = c_grad * 1
            """
            self.gradient += np.ones_like(self.data) * c.gradient

        c._backward = _backward
        return c

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._previous_nodes:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # reset gradients before backprop
        for v in topo:
            v.gradient = np.zeros_like(v.data)

        # seed gradient
        self.gradient = np.ones_like(self.data)

        # backprop in topological order
        for v in reversed(topo):
            v._backward()

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        return self + (-1 * other)

    def __rsub__(self, other):
        return other + (-1 * self)
