import numpy as np
from activations import tanh
import matplotlib.pyplot as plt
import random


class Linear:

    def __init__(self, dim_in, dim_out, act):
        self.W = self._init_weights(dim_out, dim_in)
        self.b = np.random.rand(dim_out, 1)
        self.g_x = None
        self.g_w = None
        self.g_b = None
        self.v = None
        self.act = act()

    def _init_weights(self, prev_layer, next_layer):
        return np.random.randn(prev_layer, next_layer) * np.sqrt(2 / next_layer)
        # return np.random.uniform(-1, 1, size=(prev_layer, next_layer)) * np.sqrt(6./(prev_layer + next_layer))

    def jacTMV_b(self, x, v):
        wx_b = self.W @ x + self.b
        grad_batch = np.multiply(self.act.deriv(wx_b), v)
        return np.mean(grad_batch, axis=1).reshape(self.b.shape[0], self.b.shape[-1])

    def jacTMV_w(self, x, v):
        wx_b = self.W @ x + self.b
        return 1/x.shape[-1] * np.multiply(self.act.deriv(wx_b), v) @ x.T

    def jacTMV_x(self, x, v):
        wx_b = (self.W @ x) + self.b
        act_deriv = self.act.deriv(wx_b)
        act_hadamard = np.multiply(act_deriv, v)
        return self.W.T @ act_hadamard

    def jacMV_x(self, x, v):
        wx_b = self.W @ x + self.b
        act_deriv = self.act.deriv(wx_b)
        diag_act_deriv = np.diag(act_deriv.reshape(act_deriv.shape[0],))
        diag_w = np.matmul(diag_act_deriv, self.W)
        return np.matmul(diag_w, v)

    # w: k*n
    # x: n*1
    # b: k*1
    # v: kn * 1 after raveling (k*n)
    # def jacMV_w(self, x, v):
    #     wx_b = self.W @ x + self.b
    #     act_deriv = self.act.deriv(wx_b)
    #     # act_deriv = wx_b
    #     diag_act = np.diag(act_deriv.reshape(act_deriv.shape[0],))
    #     x_kron_id = np.kron(x.T, np.eye(self.W.shape[0]))
    #     return (diag_act @ x_kron_id) @ v

    def jacMV_w(self, x, v):
        wx_b = self.W @ x + self.b
        act_deriv = self.act.deriv(wx_b)
        # act_deriv = wx_b
        diag_act = np.multiply(act_deriv, (v @ x))
        return diag_act

    def jacMV_b(self, x, v):
        act_deriv = self.act.deriv(np.add(np.matmul(self.W, x), self.b))
        diag_act_deriv = np.diag(act_deriv.reshape(act_deriv.shape[0],))
        return diag_act_deriv @ v

    def backward(self, x, v):
        self.g_x = self.jacTMV_x(x, v)
        self.g_w = self.jacTMV_w(x, v)
        self.g_b = self.jacTMV_b(x, v)
        # self.v = np.matmul(self.g_x.T, v)

    def step(self, opt):
        assert self.g_w is not None
        self.W = opt.step(self.g_w, self.W)
        self.b = opt.step(self.g_b, self.b)

    def __call__(self, x):
        # return self.W @ x + self.b
        return self.act.activate((self.W @ x) + self.b)


# def pre:
#     sample = random.sample(list(range(X.shape[1])), batch_size)
#     batch_X = X[:][sample]
#     batch_C = C[:][sample]
