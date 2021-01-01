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

    def jackTMV_b(self, x, v):
        wx_b = self.W @ x + self.b
        grad_batch = np.multiply(self.act.deriv(wx_b), v)
        return np.mean(grad_batch, axis=1).reshape(self.b.shape[0], self.b.shape[-1])

    def jackTMV_w(self, x, v):
        wx_b = self.W @ x + self.b
        res = np.multiply(self.act.deriv(wx_b), v) @ x.T
        return res

    def jackTMV_x(self, x, v):
        wx_b = (self.W @ x) + self.b
        act_deriv = self.act.deriv(wx_b)
        act_hadamard = np.multiply(act_deriv, v)
        return self.W.T @ act_hadamard

    def jackMV_x(self, x, v):
        wx_b = self.W @ x + self.b
        act_deriv = self.act.deriv(wx_b)
        diag_act_deriv = np.diag(act_deriv.reshape(act_deriv.shape[0],))
        diag_w = np.matmul(diag_act_deriv, self.W)
        return np.matmul(diag_w, v)

    def jackMV_w(self, x, v):
        wx_b = self.W @ x + self.b
        act_deriv = self.act.deriv(wx_b)
        diag_act = np.diag(act_deriv.reshape(act_deriv.shape[0],))
        x_kron_id = np.kron(x.T, np.eye(self.W.shape[-1]))
        return (diag_act @ x_kron_id) @ v

    def jackMV_b(self, x, v):
        act_deriv = self.act.deriv(np.add(np.matmul(self.W, x), self.b))
        diag_act_deriv = np.diag(act_deriv.reshape(act_deriv.shape[0],))
        return diag_act_deriv @ v

    def backward(self, x, v):
        self.g_x = self.jackTMV_x(x, v)
        self.g_w = self.jackTMV_w(x, v)
        self.g_b = self.jackTMV_b(x, v)
        # self.v = np.matmul(self.g_x.T, v)

    def step(self, opt):
        assert self.g_w is not None
        self.W = opt.step(self.g_w, self.W)
        self.b = opt.step(self.g_b, self.b)

    def __call__(self, x):
        return self.act.activate((self.W @ x) + self.b)


def jacMV_x_test():
    d = np.random.rand(3, 1)
    x = np.random.rand(3, 1)
    normalized_d = d / np.linalg.norm(d)

    eps_num = 20
    eps_vals = np.geomspace(0.5, 0.5 ** eps_num, eps_num)
    lin1 = Linear(3, 3, tanh)
    fx = lin1(x)
    no_grad, x_grad = [], []

    for eps in eps_vals:
        e_normalized_d = eps * normalized_d
        x_perturbatzia = np.add(x, e_normalized_d)
        fx_d = lin1(x_perturbatzia)
        jackMV_x = lin1.jackMV_x(x, e_normalized_d)
        print(jackMV_x)
        print('epsilon: ', eps)
        print(np.linalg.norm(np.subtract(fx_d, fx)))
        print(np.linalg.norm(np.subtract(np.subtract(fx_d, fx), jackMV_x)))
        no_grad.append(np.linalg.norm(np.subtract(fx_d, fx)))
        x_grad.append(np.linalg.norm(np.subtract(np.subtract(fx_d, fx), jackMV_x)))

    l = range(eps_num)
    plt.plot(l, no_grad, 'k', label='No gradient')
    plt.plot(l, x_grad, 'g', label='With gradient')
    plt.yscale('log')
    plt.legend()
    plt.show()


def jacMV_b_test():
    d = np.random.rand(3, 1)
    x = np.random.rand(3, 1)
    normalized_d = d / np.linalg.norm(d)

    eps_num = 20
    eps_vals = np.geomspace(0.5, 0.5 ** eps_num, eps_num)

    lin1 = Linear(3, 3, tanh)
    fx = lin1(x)

    no_grad, b_grad = [], []
    b = lin1.b

    for eps in eps_vals:

        eps_d = eps * normalized_d
        lin1.b = np.add(b, eps_d)

        fx_d = lin1(x)

        jackMV_b = lin1.jackMV_b(x, eps_d)

        no_grad.append(np.linalg.norm(np.subtract(fx_d, fx)))
        b_grad.append(np.linalg.norm(np.subtract(np.subtract(fx_d, fx), jackMV_b)))

    l = range(eps_num)
    plt.plot(l, no_grad, 'k', label='No gradient')
    plt.plot(l, b_grad, 'g', label='With gradient')
    plt.yscale('log')
    plt.legend()
    plt.show()


def jacMV_w_test():
    d = np.random.rand(3, 3)
    x = np.random.rand(3, 1)
    normalized_d = d / np.linalg.norm(d)

    eps_num = 20
    eps_vals = np.geomspace(0.5, 0.5 ** eps_num, eps_num)

    lin1 = Linear(3, 3, tanh)
    fx = lin1(x)

    no_grad, w_grad = [], []
    w = lin1.W

    for eps in eps_vals:

        eps_d = eps * normalized_d
        lin1.W = np.add(w, eps_d)

        fx_d = lin1(x)

        jackMV_w = lin1.jackMV_w(x, eps_d.ravel())

        no_grad.append(np.linalg.norm(np.subtract(fx_d, fx)))
        w_grad.append(np.linalg.norm(np.subtract(np.subtract(fx_d, fx), jackMV_w)))

    l = range(eps_num)
    plt.plot(l, no_grad, 'k', label='No gradient')
    plt.plot(l, w_grad, 'g', label='With gradient')
    plt.yscale('log')
    plt.legend()
    plt.show()


def jacTMV_w_test():
    v = np.random.rand(3, 1)
    x = np.random.rand(3, 1)
    u = np.random.rand(3, 1)
    lin1 = Linear(3, 3, tanh)

    jacMV_w = lin1.jackMV_w(x, v)

    lin1.backward(x, u)
    jacTMV_w = lin1.g_w

    u_jac = u.T @ jacMV_w
    v_jacT = v.T @ jacTMV_w

    print(abs(np.subtract(u_jac, v_jacT)))


def jacTMV_b_test():
    v = np.random.rand(3, 1)
    x = np.random.rand(3, 1)
    u = np.random.rand(3, 1)
    lin1 = Linear(3, 3, tanh)

    jackMV_b = lin1.jackMV_b(x, v)


    lin1.backward(x, u)
    jackTMV_b = lin1.g_b


    u_jack = u.T @ jackMV_b
    v_jackT = v.T @ jackTMV_b

    print(abs(np.subtract(u_jack, v_jackT)))


def jacTMV_x_test():
    v = np.random.rand(3, 1)
    x = np.random.rand(3, 1)
    u = np.random.rand(3, 1)
    lin1 = Linear(3, 3, tanh)

    jacMV_x = lin1.jackMV_x(x, v)


    lin1.backward(x, u)
    jacTMV_x = lin1.g_x


    u_jac = u.T @ jacMV_x
    v_jacT = v.T @ jacTMV_x

    print(abs(np.subtract(u_jac, v_jacT)))


# jacTMV_test()
# jacMV_x_test()
jacTMV_b_test()
jacTMV_x_test()
jacTMV_w_test()
# jacMV_w_test()
# def pre:
#     sample = random.sample(list(range(X.shape[1])), batch_size)
#     batch_X = X[:][sample]
#     batch_C = C[:][sample]
