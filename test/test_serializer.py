import os
import numpy as np

from src.ModelSerializer import ModelSerializer
from src.RNNLayer import RnnNetwork


def test_serializer():
    hidden_dim = 20
    input_dim = 3
    filename = "resources/test.model"

    rnn_original = RnnNetwork(input_dim, hidden_dim)
    serializer = ModelSerializer(filename)
    serializer.set_model(rnn_original)
    serializer.serialize()
    rnn = serializer.deserialize()
    assert np.array_equal(rnn.W, rnn_original.W)
    assert np.array_equal(rnn.U, rnn_original.U)
    assert np.array_equal(rnn.V, rnn_original.V)
    os.remove(filename)
