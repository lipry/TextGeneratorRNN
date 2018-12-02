import logging
import numpy as np

from datetime import datetime
from src.FileReader import OneHotEncodingUtilities
from src.Operations import Sigmoid, OutputLayer
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

        self.U = np.random.uniform(-np.sqrt(1. / input_dim), np.sqrt(1. / input_dim), (hidden_dim, input_dim))
        self.W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (input_dim, hidden_dim))

        self.losses = []
        self.test_losses = []

    def forward_prop(self, x):
        cells = []
        h_prev = np.zeros(self.hidden_dim)
        for elem in x:
            v = OneHotEncodingUtilities.one_hot_encoder(elem, self.input_dim)
            cell = RNNLayer()
            cell.forward_pass(v, h_prev, self.U, self.W, self.V)
            h_prev = cell.h
            cells.append(cell)
        return cells

    def backprop_through_time(self, x, y, truncated=7):
        layers = self.forward_prop(x)
        T = len(layers)
        dU = np.zeros_like(self.U)
        dW = np.zeros_like(self.W)
        dV = np.zeros_like(self.V)

        output = OutputLayer()
        prev_ht = np.zeros(self.hidden_dim)
        diff_h = np.zeros(self.hidden_dim)
        for t in range(0, T):
            diff_Vprod = output.diff(layers[t].Vproduct, y[t])
            v = OneHotEncodingUtilities.one_hot_encoder(x[t], self.input_dim)
            _, dh_prev, dUt, dWt, dVt = layers[t].backward_pass(v, prev_ht, self.U, self.W, self.V, diff_h, diff_Vprod)
            prev_ht = layers[t].h
            diff_Vprod = np.zeros(self.input_dim)
            for i in range(t-1, max(t-1-truncated, -1), -1):
                v = OneHotEncodingUtilities.one_hot_encoder(x[i], self.input_dim)
                prev_hi = layers[i].h if i != 0 else np.zeros(self.hidden_dim)
                _, dh_prev, dUi, dWi, dVi = layers[i].backward_pass(v, prev_hi, self.U, self.W, self.V, dh_prev, diff_Vprod)
                dUt += dUi
                dWt += dWi
            dU += dUt
            dW += dWt
            dV += dVt
        return np.array([dU, dW, dV])

    def batch_loss(self, x, y):
        out = OutputLayer()
        layers = self.forward_prop(x)
        loss = 0.0
        for i, layer in enumerate(layers):
            loss += out.loss(layer.Vproduct, y[i])

        return loss/float(len(x))

    # def total_loss(self, X, Y):
    #     loss = 0.0
    #     for i in range(len(Y)):
    #         loss += self.batch_loss(X[i], Y[i])
    #     return loss/float(len(X))

    def total_loss(self, X, Y):
        vect = np.vectorize(self.batch_loss, otypes=[np.float])
        return vect(X, Y).sum()/float(len(X))

    def print_loss(self, X, Y, epoch, test=False):
        loss = self.total_loss(X, Y)
        if test:
            self.test_losses.append(loss)
        else:
            self.losses.append(loss)
        time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        s = "{}: epoch = {} loss = {}"
        if test:
            s = "TEST LOSS: " + s
        logging.debug(s.format(time, epoch, loss))

    def train(self, X_train, Y_train, X_test, Y_test, epochs=100, step_size=0.001, decay_rate1=0.9, decay_rate2=0.999, eps=1e-8):
        assert len(X_train) == len(Y_train)
        for epoch in range(epochs):
            logging.debug("{}: Analyzing epoch = {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch))
            # check loss
            if (epoch % 5) == 0:
                self.print_loss(X_train, Y_train, epoch)
                self.print_loss(X_test, Y_test, epoch, test=True)
            # init first and second moment vector
            # dU, dW, dV
            m_t_V = np.zeros_like(self.V)
            m_t_W = np.zeros_like(self.W)
            m_t_U = np.zeros_like(self.U)

            v_t_V = np.zeros_like(self.V)
            v_t_W = np.zeros_like(self.W)
            v_t_U = np.zeros_like(self.U)

            t = 0
            for i in range(len(Y_train)):
                t += 1
                g_t = self.backprop_through_time(X_train[i], Y_train[i])

                m_t_V = decay_rate1*m_t_V + (1 - decay_rate1)*g_t[2]
                m_t_W = decay_rate1*m_t_W + (1 - decay_rate1)*g_t[1]
                m_t_U = decay_rate1*m_t_U + (1 - decay_rate1)*g_t[0]

                v_t_V = decay_rate2*v_t_V + (1 - decay_rate2)*(g_t[2]**2)
                v_t_W = decay_rate2*v_t_W + (1 - decay_rate2)*(g_t[1]**2)
                v_t_U = decay_rate2*v_t_U + (1 - decay_rate2)*(g_t[0]**2)

                m_correct_V = m_t_V/(1 - (decay_rate1**t))
                m_correct_W = m_t_W/(1 - (decay_rate1**t))
                m_correct_U = m_t_U/(1 - (decay_rate1**t))

                v_correct_V = v_t_V/(1 - decay_rate2**t)
                v_correct_W = v_t_W/(1 - decay_rate2**t)
                v_correct_U = v_t_U/(1 - decay_rate2**t)
                self.V -= (step_size * m_correct_V) / (np.sqrt(v_correct_V)+eps)
                self.W -= (step_size * m_correct_W) / (np.sqrt(v_correct_W)+eps)
                self.U -= (step_size * m_correct_U) / (np.sqrt(v_correct_U)+eps)

    def prediction(self, x):
        out = OutputLayer()
        layers = self.forward_prop(x)
        return [np.argmax(out.predict(layer.Vproduct)) for layer in layers]
