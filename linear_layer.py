import numpy as np
from activations import tanh
import random


class Linear:

    def __init__(self, dim_in, dim_out, act):
        self.W = np.random.rand(dim_out, dim_in)
        self.b = np.random.rand(dim_out, 1)
        self.g_x = None
        self.g_w = None
        self.g_b = None
        self.v = None
        self.act = act()

    def grad_b(self, x, v):
        return np.multiply(self.act.deriv((np.add(np.matmul(self.W, x), self.b))), v)

    def grad_W(self, x, v):
        return np.matmul(np.multiply(self.act.deriv(np.add(np.matmul(self.W, x), self.b)), v), x.T)

    def grad_x(self, x, v):
        act_deriv = self.act.deriv(np.add(np.matmul(self.W, x), self.b))
        act_hadamard = np.multiply(act_deriv, v)
        return np.matmul(self.W.T, act_hadamard)

    def jackMV(self, x, v):
        act_deriv = self.act.deriv(np.add(np.matmul(self.W, x), self.b))
        diag_act_deriv = np.diag(act_deriv.reshape(act_deriv.shape[0],))
        diag_w = np.matmul(diag_act_deriv, self.W)
        return np.matmul(diag_w, v)

    def backward(self, x, v):
        self.g_x = self.grad_x(x, v)
        self.g_w = self.grad_W(x, v)
        self.g_b = self.grad_b(x, v)
        self.v = np.matmul(self.g_x.T, v)

    def sgd(self, grad, param, lr=0.001):
        return np.subtract(param, lr * grad)

    def step(self):
        assert self.g_w is not None
        self.W = self.sgd(self.g_w, self.W)
        self.b = self.sgd(self.g_b, self.b)

    def forward(self, x):
        return self.act.activate(np.add(np.matmul(self.W, x), self.b))


def jacMV_test():
    d = np.random.rand(3, 1)
    x = np.random.rand(3, 1)
    normalized_d = d / np.linalg.norm(d)
    epsilon = [1e-2, 1e-3, 1e-4, 1e-5]
    lin1 = Linear(3, 3, tanh)
    fx = lin1.forward(x)
    for e in epsilon:
        e_normalized_d = e * normalized_d
        x_perturbatzia = np.add(x, e_normalized_d)
        fx_d = lin1.forward(x_perturbatzia)
        jackMV_ = lin1.jackMV(x, e_normalized_d)
        print(jackMV_)
        print('epsilon: ', e)
        print(np.linalg.norm(np.subtract(fx_d, fx)))
        print(np.linalg.norm(np.subtract(np.subtract(fx_d, fx), jackMV_)))


def jacTMV_test():
    v = np.random.rand(3, 1)
    x = np.random.rand(3, 1)
    u = np.random.rand(3, 1)
    lin1 = Linear(3, 3, tanh)
    jackMV_ = lin1.jackMV(x, v)
    print(jackMV_)
    lin1.backward(x, u)
    jackTMV_ = lin1.g_x
    print(jackTMV_)
    u_jack = np.matmul(u.T, jackMV_)
    v_jackT = np.matmul(v.T, jackTMV_)
    print(abs(np.subtract(u_jack, v_jackT)))



jacTMV_test()
# def pre:
#     sample = random.sample(list(range(X.shape[1])), batch_size)
#     batch_X = X[:][sample]
#     batch_C = C[:][sample]
