import numpy as np


class tanh:
    def __init__(self):
        self.activate = np.tanh
        self.deriv = lambda x: np.subtract(np.ones(x.shape), np.multiply(np.tanh(x), np.tanh(x)))
