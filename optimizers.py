import numpy as np


def sgd(grad, param, lr=0.001):
    return np.subtract(param, lr * grad)
