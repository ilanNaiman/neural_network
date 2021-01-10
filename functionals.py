import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib as npm
from sklearn.utils import shuffle
from data_loader import loadGMMData
from tqdm import tqdm


class CrossEntropy:

    def __init__(self, W):
        self.grad_W = None
        self.W = W

    def grad_w(self, X, C):
        """
        :param X: dim(L-1) * m (m number of examples, n dimension of each exmaple)
        :param W: dim(L-1) * nlabels
        :param C: m * nlables
        :return: grad dim(L-1) * nlabels
        """
        m = X.shape[-1]

        # use for all calculations
        x_w = np.exp(X.T @ self.W) # m * nlabels
        stacked_x_w = np.array(x_w.sum(axis=1)) # m * 1
        diag = np.linalg.inv(np.diag(stacked_x_w)) # m * m diag
        diag_exp = diag @ x_w # m * nlabels
        # diag_exp = x_w / stacked_x_w
        e = np.subtract(diag_exp, C) # m * nlabels
        self.grad_W = 1/m * np.matmul(X, e)

    def grad_inp(self, X, C):
        """
        :self.W : dim(L-1) * nlabels weights
        :param X: dim(L-1) * m (m number of examples, n dimension of each exmaple)
        :param W : dim(L-1) * nlabels
        :param C: m * nlables
        :return: dim(L-1) * nlabels
        """
        m = X.shape[-1]

        # use for all calculations
        w_x = np.exp(np.matmul(self.W.T, X)) # n * m
        stacked_x_w = np.array(w_x.sum(axis=0)) # 1 * m
        rep_w_x = npm.repmat(stacked_x_w, w_x.shape[0], 1) # n * m
        div_w_x = np.divide(w_x, rep_w_x) # n * m
        subc_w_x = np.subtract(div_w_x, C.T) # n * m
        return 1/m * np.matmul(self.W, subc_w_x) # d * m

    def __call__(self, X, C):
        """
        :param X: dim(L-1) * m (m number of examples, n dimension of each exmaple)
        :param W : dim(L-1) * nlabels
        :param C: m * nlables
        :return: scalar
        """
        m = X.shape[-1]

        x_w = np.exp(X.T @ self.W) # m * nlabels
        stacked_x_w = np.array(x_w.sum(axis=1)) # m * 1
        diag = np.linalg.inv(np.diag(stacked_x_w)) # m * m diag
        log_out = np.log(diag @ x_w) # m * nlabels
        c_log_out = np.trace(C.T @ log_out)

        return -1/m * c_log_out

    def step(self, opt):
        opt.step(self.grad_W, self.W)


class Softmax:

    def __init__(self, dim_in, dim_out):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.W = self._init_weights(dim_in, dim_out)

    def _init_weights(self, prev_layer, next_layer):
        return np.random.randn(prev_layer, next_layer) * np.sqrt(2 / next_layer)

    def __call__(self, X):
        """
        self.W: W -> dim(L-1) * nlabels weights
        :param X: dim(L-1) * m (m number of examples, n dimension of each exmaple)
        :return: vector of distribution over the labels prediction
        """
        linear = X.T @ self.W   # m * nlabels
        exp_active = np.exp(linear)
        sums = np.sum(exp_active, axis=1).reshape(-1, 1)  # column of sums
        softmax_res = exp_active / sums
        return softmax_res
