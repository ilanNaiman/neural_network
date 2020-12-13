import numpy as np
import random


# X -> dim(L-1) * m (m number of examples, n dimension of each exmaple)
# W -> dim(L-1) * nlabels weights
# C -> m * nlables
# output -> dim(L-1) * nlabels
def softmax_grad_w(X, W, C):
    m = X.shape[0]

    # use for all calculations
    x_w = np.exp(np.matmul(X.transpose(), W)) # m * nlabels
    stacked_x_w = np.array(x_w.sum(axis=1)) # m * 1
    diag = np.linalg.inv(np.diag(stacked_x_w)) # m * m diag
    diag_exp = np.matmul(diag, x_w) # m * nlabels
    e = np.subtract(diag_exp, C) # m * nlabels
    return 1/m * np.matmul(X, e)


# X -> dim(L-1) * m (m number of examples, n dimension of each exmaple)
# W -> dim(L-1) * nlabels weights
# C -> m * nlables
# output -> dim(L-1) * nlabels
def softmax_grad_inp(X, W, C):
    m = X.shape[0]

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


def softmax_n(X, W, C):
    m = X.shape[0]
    x_w = np.exp(np.matmul(X.transpose(), W)) # m * nlabels
    stacked_x_w = np.array(x_w.sum(axis=1)) # m * 1
    diag = np.linalg.inv(np.diag(stacked_x_w))  # m * m diag
    log_out = np.log(np.matmul(diag, x_w))
    c_log_out = np.trace(np.matmul(C.T, log_out))
    return -1/m * c_log_out


#
def grad_test():
    d = np.random.rand(3, 1)
    x = np.random.rand(3, 1)
    c = np.asarray([[1]])
    w = np.random.rand(3, 1)
    normalized_d = d / np.linalg.norm(d)
    epsilon = [1e-2, 1e-3, 1e-4, 1e-5]
    fx = softmax_n(x, w, c)
    for e in epsilon:
        e_normalized_d = e * normalized_d
        x_perturbatzia = np.add(x, e_normalized_d)
        fx_d = softmax_n(x_perturbatzia, w, c)
        softmax_grad_ret = softmax_grad_w(x, w, c)
        print('epsilon: ', e)
        print(abs(fx_d - fx))
        print(abs(fx_d - fx - e * np.matmul(normalized_d.transpose(), softmax_grad_ret)))


# grad_test()
# sgd(softmax_grad, np.eye(4), np.random.rand(3, 6), np.eye(4), batch_size=2)
