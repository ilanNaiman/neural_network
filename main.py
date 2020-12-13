import numpy as np


def forward(W, x, b):
    return tanh(linear(W, x, b))


def linear(W, x, b):
    return np.add(np.matmul(W, x), b)


def tanh(h):
    return np.tanh(h)


