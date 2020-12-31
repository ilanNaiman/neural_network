import numpy as np


class SGD:
    def __init__(self, lr=0.001):

        self.lr = lr

    def step(self, grad, param):
        return np.subtract(param, self.lr * grad)
