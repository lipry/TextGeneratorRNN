import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class RNN:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        matrix_columns = hidden_dim + input_dim

        self.Wg = np.random.rand(hidden_dim, matrix_columns)
        self.Wi = np.random.rand(hidden_dim, matrix_columns)
        self.Wf = np.random.rand(hidden_dim, matrix_columns)
        self.Wo = np.random.rand(hidden_dim, matrix_columns)

        self.bg = np.random.rand(hidden_dim)
        self.bi = np.random.rand(hidden_dim)
        self.bf = np.random.rand(hidden_dim)
        self.bo = np.random.rand(hidden_dim)

        self.DWg = np.zeros_like(self.Wg)
        self.DWi = np.zeros_like(self.Wi)
        self.DWf = np.zeros_like(self.Wf)
        self.DWo = np.zeros_like(self.Wo)

    def forward_pass(self, x, h_prev, c_prev):
        # TODO: TESTING
        x_h = np.hstack((x, h_prev))

        f = sigmoid(self.Wf.dot(x_h) + self.bf)
        i = sigmoid(self.Wi.dot(x_h) + self.bi)
        o = sigmoid(self.Wo.dot(x_h) + self.bo)
        g = np.tanh(self.Wg.dot(x_h) + self.bg)

        c = f*c_prev + i*g
        h = o * np.tanh(c)

        return c, h
