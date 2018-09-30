from src.RNN import RNN
import numpy as np


def test_init():
    hidden_dim = 20
    input_dim = 3
    rnn = RNN(input_dim, hidden_dim)
    init_weights_matrix(rnn, hidden_dim, input_dim)
    init_bias_vectors(rnn, hidden_dim)
    init_differential_matrix(rnn, hidden_dim, input_dim)


def init_weights_matrix(rnn, hidden_dim, input_dim):
    assert rnn.Wg.shape == (hidden_dim, input_dim+hidden_dim)
    assert rnn.Wi.shape == (hidden_dim, input_dim+hidden_dim)
    assert rnn.Wf.shape == (hidden_dim, input_dim+hidden_dim)
    assert rnn.Wo.shape == (hidden_dim, input_dim+hidden_dim)

    assert (rnn.Wg >= 0).all() and (rnn.Wg <= 1).all()
    assert (rnn.Wi >= 0).all() and (rnn.Wi <= 1).all()
    assert (rnn.Wf >= 0).all() and (rnn.Wf <= 1).all()
    assert (rnn.Wo >= 0).all() and (rnn.Wo <= 1).all()


def init_bias_vectors(rnn, hidden_dim):
    assert rnn.bg.shape == (hidden_dim, )
    assert rnn.bi.shape == (hidden_dim, )
    assert rnn.bf.shape == (hidden_dim, )
    assert rnn.bo.shape == (hidden_dim, )

    assert (rnn.bg >= 0).all() and (rnn.bg <= 1).all()
    assert (rnn.bi >= 0).all() and (rnn.bi <= 1).all()
    assert (rnn.bf >= 0).all() and (rnn.bf <= 1).all()
    assert (rnn.bo >= 0).all() and (rnn.bo <= 1).all()


def init_differential_matrix(rnn, hidden_dim, input_dim):
    assert rnn.DWg.shape == (hidden_dim, input_dim+hidden_dim)
    assert rnn.DWi.shape == (hidden_dim, input_dim+hidden_dim)
    assert rnn.DWf.shape == (hidden_dim, input_dim+hidden_dim)
    assert rnn.DWo.shape == (hidden_dim, input_dim+hidden_dim)

    assert (rnn.DWg == 0).all()
    assert (rnn.DWi == 0).all()
    assert (rnn.DWf == 0).all()
    assert (rnn.DWo == 0).all()


def test_forward_pass():
    hidden_dim = 1
    input_dim = 2
    rnn = RNN(input_dim, hidden_dim)
    rnn.Wg = np.array([[0.45, 0.25, 0.15]])
    rnn.Wi = np.array([[0.95, 0.8, 0.8]])
    rnn.Wf = np.array([[0.7, 0.45, 0.1]])
    rnn.Wo = np.array([[0.6, 0.4, 0.25]])

    rnn.bg = np.array([0.2])
    rnn.bi = np.array([0.65])
    rnn.bf = np.array([0.15])
    rnn.bo = np.array([0.1])
    # T = 0
    x0 = np.array([1, 2])
    c1, h1 = rnn.forward_pass(x0, np.array([0]), np.array([0]))

    assert np.allclose(c1, np.array([0.78572]))
    assert np.allclose(h1, np.array([0.53631]))
    # T = 1
    x1 = np.array([0.5, 3])
    c2, h2 = rnn.forward_pass(x1, h1, c1)

    assert np.allclose(c2, np.array([1.51763]))
    assert np.allclose(h2, np.array([0.77198]))




