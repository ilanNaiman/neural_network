import matplotlib.pyplot as plt
import numpy as np
import random


# X -> dim(L-1) * m (m number of examples, n dimension of each exmaple)
# W -> dim(L-1) * nlabels weights
# C -> m * nlables
# output -> dim(L-1) * nlabels
def softmax_grad_w(X, W, C):
    m = X.shape[-1]

    # use for all calculations
    x_w = np.exp(X.T @ W) # m * nlabels
    stacked_x_w = np.array(x_w.sum(axis=1)) # m * 1
    diag = np.linalg.inv(np.diag(stacked_x_w)) # m * m diag
    diag_exp = diag @ x_w # m * nlabels
    e = np.subtract(diag_exp, C) # m * nlabels
    return 1/m * np.matmul(X, e)


# X -> dim(L-1) * m (m number of examples, n dimension of each exmaple)
# W -> dim(L-1) * nlabels weights
# C -> m * nlables
# output -> dim(L-1) * nlabels
def softmax_grad_inp(X, W, C):
    m = X.shape[-1]

    # use for all calculations
    w_x = np.exp(np.matmul(W.T, X)) # n * m
    stacked_x_w = np.array(w_x.sum(axis=0)) # 1 * m
    rep_w_x = np.matlib.repmat(stacked_x_w, w_x.shape[0], 1) # n * m
    div_w_x = np.divide(w_x, rep_w_x) # n * m
    subc_w_x = np.subtract(div_w_x, C.T) # n * m
    return 1/m * np.matmul(W, subc_w_x) # d * m


def sgd(grad_f, X, W, C, lr=0.001, batch_size=1):
    sample = random.sample(list(range(X.shape[1])), batch_size)
    batch_X = X[:][sample]
    batch_C = C[:][sample]
    grad_w = grad_f(batch_X, W, batch_C)
    updated_W = np.subtract(W, lr * grad_w)
    return updated_W


# C -> m * nlables
def softmax_apply(X, W, C):

    m = X.shape[-1]

    x_w = np.exp(X.T @ W) # m * nlabels
    stacked_x_w = np.array(x_w.sum(axis=1)) # m * 1
    diag = np.linalg.inv(np.diag(stacked_x_w)) # m * m diag
    log_out = np.log(diag @ x_w) # m * nlabels
    c_log_out = np.trace(C.T @ log_out)

    return -1/m * c_log_out


#
def grad_test():
    d = np.random.rand(5, 3)
    x = np.random.rand(5, 2)
    c = np.random.rand(2, 3)
    w = np.random.rand(5, 3)
    normalized_d = d / np.linalg.norm(d)
    epsilon = np.geomspace(0.5, 0.5 ** 30 , 30)
    fx = softmax_apply(x, w, c)                     # f(w)
    softmax_grad_ret = softmax_grad_w(x, w, c)      # grad(w)
    no_grad, w_grad = [], []
    for e in epsilon:
        e_normalized_d = e * normalized_d           # e * d
        w_perturbatzia = np.add(w, e_normalized_d)  # w + e*d
        fx_d = softmax_apply(x, w_perturbatzia, c)  # f(w + e*d)
        normalized_d_r = normalized_d.ravel()       # d
        print(np.linalg.norm(normalized_d_r))
        softmax_grad_ret_r = softmax_grad_ret.ravel()
        print('epsilon: ', e)
        no_grad.append(abs(fx_d - fx))
        w_grad.append(abs(fx_d - fx - e * normalized_d_r.T @ softmax_grad_ret_r))
        print('fx_d - fx:', abs(fx_d - fx))
        print('fx_d - fx - grad: ', abs(fx_d - fx - e * normalized_d_r.T @ softmax_grad_ret_r))

    plt.plot(epsilon, no_grad, 'k', label='No gradient')
    plt.plot(epsilon, w_grad, 'g', label='With gradient')
    plt.yscale('log')
    plt.legend()
    plt.show()

grad_test()
# sgd(softmax_grad, np.eye(4), np.random.rand(3, 6), np.eye(4), batch_size=2)
