import numpy as np

from src.RNNLayer import RnnNetwork, RNNLayer

hidden_dim = 20
input_dim = 3
rnn = RnnNetwork(input_dim, hidden_dim)

U = np.array([[0.45, 0.25], [0.12, 0.54]])
W = np.array([[0.95, 0.80], [0.12, 0.21]])
V = np.array([[0.70, 0.45], [0.90, 0.15]])


def test_init():
    check_init_weights_matrix()


def check_init_weights_matrix():
    assert rnn.U.shape == (hidden_dim, input_dim)
    assert rnn.W.shape == (hidden_dim, hidden_dim)
    assert rnn.V.shape == (input_dim, hidden_dim)

    assert (rnn.W >= 0).all() and (rnn.W <= 1).all()
    assert (rnn.U >= 0).all() and (rnn.U <= 1).all()
    assert (rnn.V >= 0).all() and (rnn.V <= 1).all()


def test_rnn_layer_forward_pass():
    rnncell = RNNLayer()

    # T = 0
    x0 = np.array([0, 1])
    h_prev = np.array([0, 0])
    rnncell.forward_pass(x0, h_prev, U, W, V)

    assert np.allclose(rnncell.h, np.array([0.24491866, 0.49298797]))

    # T = 1
    x1 = np.array([1, 0])
    h_prev = rnncell.h
    rnncell.forward_pass(x1, h_prev, U, W, V)

    assert np.allclose(rnncell.h, np.array([0.79210745, 0.24765939]))


def test_rnnnetwork_forward_propagation():
    x = np.array([1, 2])
    cells = rnn.forward_prop(x)
    assert len(cells) == 2


def test_rnnnetwork_backpropagation():
    x = np.array([1, 2])
    y = np.array([2, 1])
    dU, dW, dV = rnn.backprop_through_time(x, y)
    assert dU.shape == (hidden_dim, input_dim)
    assert dW.shape == (hidden_dim, hidden_dim)
    assert dV.shape == (input_dim, hidden_dim)

    assert (dW >= -1).all() and (dW <= 1).all()
    assert (dU >= -1).all() and (dU <= 1).all()
    assert (dV >= -1).all() and (dV <= 1).all()
