import numpy as np

from src.Operations import Sigmoid
from src.Operations import Multiply
from src.Operations import Add
from src.Operations import Tanh

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


class RnnNetwork:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.U = np.random.rand(self.hidden_dim, self.input_dim)
        self.W = np.random.rand(self.hidden_dim, self.hidden_dim)
        self.V = np.random.rand(self.input_dim, self.hidden_dim)

    def forward_prop(self, x):
        cells = []
        h_prev = np.zeros(self.hidden_dim)
        for elem in x:
            cell = RNNLayer()
            cell.forward_pass(elem, h_prev, self.U, self.W, self.V)
            h_prev = cell.h
            cells.append(cell)
        return cells

    def back_prop(self, x, y):
        pass





