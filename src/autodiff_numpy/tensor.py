import numpy as np


class Tensor:
    def __init__(self, data, requires_grad=False):
        """
        A tensor class that wraps a numpy array.

        Parameters:
        data (np.ndarray): The data to wrap.
        requires_grad (bool): Whether the tensor requires gradient.
        """
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self._parents = []
        self._backward = lambda: None
        self._op = None

    def zero_grad(self):
        """
        Zeros all the gradients of the tensor.
        """
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)
