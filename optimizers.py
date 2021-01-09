import numpy as np


class SGD:
    def __init__(self, lr=0.001):

        self.lr = lr

    def step(self, grad, param):
        return np.subtract(param, self.lr * grad)


class Momentum:
    def __init__(self, gamma=0.7, lr=0.001):

        self.lr = lr
        self.gamma = gamma
        self.m = 0

    def step(self, grad, param):
        self.m = self.gamma * self.m + self.lr * grad
        return param - self.m
