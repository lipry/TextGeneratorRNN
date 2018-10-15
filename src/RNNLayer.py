import numpy as np

from src.Operations import Sigmoid
from src.Operations import Multiply
from src.Operations import Add
from src.Operations import Tanh


def mean_squared_error(pred, y):
    return np.mean((y-pred) ** 2)


class RnnNetwork:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.W = np.random.rand(self.hidden_dim, self.input_dim)
        self.U = np.random.rand(self.hidden_dim, self.input_dim)
        self.V = np.random.rand(self.hidden_dim, self.input_dim)

    def forward_pass(self, x, h_prev, c_prev):
        pass


sigmoid = Sigmoid()
add = Add()
mul = Multiply()
tanh = Tanh()


class RNNLayer:
    def __init__(self):
        self.Uproduct = None
        self.Wproduct = None
        self.UWsum = None
        self.h = None
        self.Vproduct = None

    def forward_pass(self, x, h_prev, U, W, V):
        # calculating the state function h
        self.Uproduct = mul.forward_pass(U, x)
        self.Wproduct = mul.forward_pass(W, h_prev)
        self.UWsum = add.forward_pass(self.Uproduct, self.Wproduct)
        self.h = tanh.forward_pass(self.UWsum)

        self.Vproduct = mul.forward_pass(V, self.h)

    def backward_pass(self, x, h_prev, U, W, V, diffh, dVproduct):
        self.forward_pass(x, h_prev, U, W, V)
        dV, dhv = mul.backward_pass(V, self.h, dVproduct)
        dh = diffh + dhv
        dUWsum = tanh.backward_pass(self.UWsum, dh)
        dUproduct, dWproduct = add.backward_pass(self.Uproduct, self.Wproduct, dUWsum)
        dU, dx = mul.backward_pass(U, x, dUproduct)
        dW, dh_prev = mul.backward_pass(W, h_prev, dWproduct)
        return dx, dh_prev, dU, dW, dV






