import numpy as np

from autodiff_numpy.ops import relu
from autodiff_numpy.tensor import Tensor


def test_relu_backward():
    x = Tensor(np.array([-1.0, 2.0]), requires_grad=True)
    y = relu(x).sum()
    y.backward()
    assert np.allclose(x.grad, np.array([0.0, 1.0]))
