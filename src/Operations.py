import warnings

import numpy as np


class Sigmoid:
    """Implement the forward and backward pass for a standard logistic function"""
    def forward_pass(self, x):
        return 1 / (1 + np.exp(-x))

    def backward_pass(self, x, top_diff):
        fw = self.forward_pass(x)
        return (1.0 - fw)*fw * top_diff


class Multiply:
    """Implement the forward and backward pass for the dot product"""
    def forward_pass(self, W, x):
        return W.dot(x)

    def backward_pass(self, W, x, dz):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dW = np.asarray(np.dot(np.transpose(np.asmatrix(dz)), np.asmatrix(x)))
        dx = np.dot(np.transpose(W), dz)
        return dW, dx


class Add:
    def forward_pass(self, x, y):
        return x + y

    def backward_pass(self, x, y, dz):
        return np.ones_like(x)*dz, np.ones_like(y)*dz


class Tanh:
    def forward_pass(self, x):
        return np.tanh(x)

    def backward_pass(self, x, top_diff):
        fw = self.forward_pass(x)
        return (1.0 - np.square(fw)) * top_diff


class OutputLayer:
    def predict(self, x):
        exp = np.exp(x-np.max(x))
        return exp / exp.sum(axis=0)

    def loss(self, x, y):
        p = self.predict(x)
        return -np.log(p[y])

    def diff(self, x, y):
        p = self.predict(x)
        p[y] -= 1
        return p
