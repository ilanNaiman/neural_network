import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from data_loader import loadGMMData


# X -> dim(L-1) * m (m number of examples, n dimension of each exmaple)
# W -> dim(L-1) * nlabels weights
# C -> m * nlables
# output -> dim(L-1) * nlabels
def cross_entropy_grad_w(X, W, C):
    m = X.shape[-1]

    # use for all calculations
    x_w = np.exp(X.T @ W) # m * nlabels
    stacked_x_w = np.array(x_w.sum(axis=1)) # m * 1
    diag = np.linalg.inv(np.diag(stacked_x_w)) # m * m diag
    diag_exp = diag @ x_w # m * nlabels
    # diag_exp = x_w / stacked_x_w
    e = np.subtract(diag_exp, C) # m * nlabels
    return 1/m * np.matmul(X, e)


# X -> dim(L-1) * m (m number of examples, n dimension of each exmaple)
# W -> dim(L-1) * nlabels weights
# C -> m * nlables
# output -> dim(L-1) * nlabels
def cross_entropy_grad_inp(X, W, C):
    m = X.shape[-1]

    # use for all calculations
    w_x = np.exp(np.matmul(W.T, X)) # n * m
    stacked_x_w = np.array(w_x.sum(axis=0)) # 1 * m
    rep_w_x = np.matlib.repmat(stacked_x_w, w_x.shape[0], 1) # n * m
    div_w_x = np.divide(w_x, rep_w_x) # n * m
    subc_w_x = np.subtract(div_w_x, C.T) # n * m
    return 1/m * np.matmul(W, subc_w_x) # d * m


def sgd(grad_f, X, W, C, lr=0.001):
    # sample = random.sample(list(range(X.shape[1])), batch_size)
    # batch_X = X[:][sample]
    # batch_C = C[:][sample]
    grad_w = grad_f(X, W, C)
    updated_W = np.subtract(W, lr * grad_w)
    return updated_W


# C -> m * nlables
def cross_entropy_loss_apply(X, W, C):

    m = X.shape[-1]

    x_w = np.exp(X.T @ W) # m * nlabels
    stacked_x_w = np.array(x_w.sum(axis=1)) # m * 1
    diag = np.linalg.inv(np.diag(stacked_x_w)) # m * m diag
    log_out = np.log(diag @ x_w) # m * nlabels
    c_log_out = np.trace(C.T @ log_out)

    return -1/m * c_log_out


# X -> dim(L-1) * m (m number of examples, n dimension of each exmaple)
# W -> dim(L-1) * nlabels weights
def softmax(X, W):
    linear = X.T @ W   # m * nlabels
    exp_active = np.exp(linear)
    sums = np.sum(exp_active, axis=1).reshape(-1, 1)  # column of sums
    softmax_res = exp_active / sums
    return softmax_res


def softmax_predict(X, W):
    softmax_res = softmax(X, W)
    max_args = np.argmax(softmax_res, axis=1).reshape(-1, 1)
    return max_args


#
def grad_test():
    d = np.random.rand(5, 3)
    x = np.random.rand(5, 2)
    c = np.random.rand(2, 3)
    c[1,:] = 1 - c[0,:]
    print(c)
    w = np.random.rand(5, 3)
    normalized_d = d / np.linalg.norm(d)
    epsilon = np.geomspace(0.5, 0.5 ** 20, 20)
    i = range(20)
    fx = cross_entropy_loss_apply(x, w, c)                     # f(w)
    softmax_grad_ret = cross_entropy_grad_w(x, w, c)   # grad(w)
    no_grad, w_grad = [], []
    for e in epsilon:
        e_normalized_d = e * normalized_d           # e * d
        w_perturbatzia = np.add(w, e_normalized_d)  # w + e*d
        fx_d = cross_entropy_loss_apply(x, w_perturbatzia, c)  # f(w + e*d)
        normalized_d_r = normalized_d.ravel()       # d
        # print(np.linalg.norm(normalized_d_r))
        softmax_grad_ret_r = softmax_grad_ret.ravel()
        print('epsilon: ', e)
        no_grad.append(abs(fx_d - fx))
        w_grad.append(abs(fx_d - fx - e * normalized_d_r.T @ softmax_grad_ret_r))
        print('fx_d - fx:', abs(fx_d - fx))
        print('fx_d - fx - grad: ', abs(fx_d - fx - e * normalized_d_r.T @ softmax_grad_ret_r))

    plt.plot(i, no_grad, 'k', label='No gradient')
    plt.plot(i, w_grad, 'g', label='With gradient')
    plt.yscale('log')
    plt.legend()
    plt.show()


def sgd_test():

    Xtrain, Xtest, Ytrain, Ytest = loadGMMData()

    lr = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    x = np.geomspace(2, 2 ** 8, 8)
    Xtrain, Ytrain = shuffle(Xtrain, Ytrain, random_state=2021)

    # split the data into train sets of different mini batch sizes
    train_sets = [(np.array_split(Xtrain, Xtrain.shape[-1] / i, axis=1),
                   np.array_split(Ytrain, Ytrain.shape[-1] / i, axis=1)) for i in x]
    test_sets = [(np.array_split(Xtest, Xtest.shape[-1] / i, axis=1),
                  np.array_split(Ytest, Ytest.shape[-1] / i, axis=1)) for i in x]

    epochs = 60

    W = np.random.randn(Ytrain.shape[0], 5) * np.sqrt(2 / 5)

    all_batches, all_labels = train_sets[2]
    # train loop
    loss = 0

    for e in range(epochs):
        acc_train_err = []
        loss_l = []
        for batch, labels in zip(all_batches, all_labels):

            W = sgd(cross_entropy_grad_w, batch, W, labels.T)

            loss = cross_entropy_loss_apply(batch, W, labels.T)
            loss_l.append(loss)


            # calculate train error
            labels = np.asarray([np.where(l == 1)[0][0] for l in labels.T])
            prediction = np.asarray([p[0] for p in softmax_predict(batch, W)])

            acc_train_err = np.append(acc_train_err, prediction == labels, axis=0)

        print('epcho {} train acc: {}  train loss: {}'.format(e, np.mean(acc_train_err), np.mean(loss_l)))

    # test loop
    acc_test_err = []
    all_batches, all_labels = test_sets[2]
    for batch, labels in zip(all_batches, all_labels):

        # calculate train error
        # prediction = softmax_predict(batch, W)

        labels = np.asarray([np.where(l == 1)[0][0] for l in labels.T])
        prediction = np.asarray([p[0] for p in softmax_predict(batch, W)])

        acc_test_err = np.append(acc_test_err, prediction == labels, axis=0)

    print('test error: {}'.format(np.mean(acc_test_err)))


# sgd_test()
grad_test()
# x = np.random.rand(5, 4)
# w = np.random.rand(5, 3)
# softmax_predict(x, w)
# sgd(softmax_grad, np.eye(4), np.random.rand(3, 6), np.eye(4), batch_size=2)
