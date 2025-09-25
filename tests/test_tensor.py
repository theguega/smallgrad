import numpy as np

from autodiff_numpy.tensor import Tensor


def test_simple_backward():
    x = Tensor(np.array([2.0]), requires_grad=True)
    y = x * x  # y = x^2
    y.backward()
    assert np.allclose(x.grad, np.array([4.0]))
