import numpy as np

from src.Operations import Sigmoid, Multiply, Add, Tanh, OutputLayer

sig = Sigmoid()


def test_sigmoid_forward_pass():
    z = np.array([-2, -4, 0, 2, 4])
    expected = np.array([0.119, 0.018, 0.5, 0.881, 0.982])
    assert np.allclose(sig.forward_pass(z), expected, atol=1e-03)

    # check if extreme value are close to 0.0 or 1.0
    z = np.array([-80, 80])
    expected = np.array([0.0, 1.0])
    assert np.all(abs(expected-sig.forward_pass(z)) < 1.5 * 10**(-30))


def test_sigmoid_backward_pass():
    z = np.array([-80, -2, -4, 0, 2, 4, 80])
    expected = np.array([0.0, 0.104839, 0.017676, 0.25, 0.104839, 0.017676, 0.0])
    assert np.allclose(sig.backward_pass(z, 1), expected, atol=1e-03)


mul = Multiply()


def test_multiply_forward_pass():
    A = np.array([[1, 0], [0, 1]])
    x = np.array([1, 3])
    assert np.allclose(mul.forward_pass(A, x), [1, 3])

    A = np.array([[0.5, 0.8], [0.113, 0.15]])
    x = np.array([0.3, 0.2])
    assert np.allclose(mul.forward_pass(A, x), [0.31, 0.0639])


def test_multiply_backward_pass():
    A = np.array([[1, 0], [0, 1]])
    x = np.array([1, 3])
    dA, dx = mul.backward_pass(A, x, 1)
    assert np.allclose(dA, [1, 3])
    assert np.allclose(dx, [[1, 0], [0, 1]])

    A = np.array([[0.5, 0.8], [0.113, 0.15]])
    x = np.array([0.3, 0.2])
    dA, dx = mul.backward_pass(A, x, [[2, 2], [2, 2]])
    assert np.allclose(dx, [[1.226, 1.226], [1.9, 1.9]])
    assert np.allclose(dA, [1., 1.])


add = Add()


def test_add_forward_pass():
    a = np.array([2, 1, 3, 4])
    b = np.array([3, 6, 5, 2])
    assert np.allclose(add.forward_pass(a, b), [5, 7, 8, 6])
    a = np.array([0.2, 0.52, 0.4, 0.45])
    b = np.array([0.1, 0.45, 0.3, 0.32])
    assert np.allclose(add.forward_pass(a, b), [0.3, 0.97, 0.7, 0.77])


def test_add_backward_pass():
    a = np.array([2, 1, 3, 4])
    b = np.array([3, 6, 5, 2])
    da, db = add.backward_pass(a, b, np.array([1., 1., 1., 1.]))
    assert np.allclose(da, [1., 1., 1., 1.])
    assert np.allclose(db, [1., 1., 1., 1.])
    da, db = add.backward_pass(a, b, np.array([2., 1., 2., 1.]))
    assert np.allclose(da, [2., 1., 2., 1.])
    assert np.allclose(db, [2., 1., 2., 1.])


tanh = Tanh()


def test_tanh_forward_pass():
    x = np.array([-10, -3, -2, 0, 1, 2, 10])
    expected = np.array([-1.0, -0.995, -0.96, 0, 0.761, 0.964, 1.0])
    assert np.allclose(tanh.forward_pass(x), expected, atol=1e-02)


def test_tanh_backward_pass():
    x = np.array([-3, -2, 0, 1, 2])
    expected = np.array([0.0098, 0.0765, 1, 0.419, 0.070])
    assert np.allclose(tanh.backward_pass(x, 1), expected, atol=1e-02)

    z = np.array([-10, 10])
    expected = np.array([0.0, 0.0])
    assert np.all(abs(expected-tanh.backward_pass(z, 1)) < 1.5 * 10**(-8))


out = OutputLayer()


def test_outputlayer_predict():
    x = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
    expected = np.array([0.02364054, 0.06426166, 0.1746813, 0.474833, 0.02364054, 0.06426166, 0.1746813])
    assert np.allclose(out.predict(x), expected)