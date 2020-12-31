import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib as npm
from sklearn.utils import shuffle
from data_loader import loadGMMData
from tqdm import tqdm


class CrossEntropy:

    def __init__(self):
        self.grad_W = None
        return

    def grad_w(self, X, W, C):
        """
        :param X: dim(L-1) * m (m number of examples, n dimension of each exmaple)
        :param W : dim(L-1) * nlabels
        :param C: m * nlables
        :return: grad dim(L-1) * nlabels
        """
        m = X.shape[-1]

        # use for all calculations
        x_w = np.exp(X.T @ W) # m * nlabels
        stacked_x_w = np.array(x_w.sum(axis=1)) # m * 1
        diag = np.linalg.inv(np.diag(stacked_x_w)) # m * m diag
        diag_exp = diag @ x_w # m * nlabels
        # diag_exp = x_w / stacked_x_w
        e = np.subtract(diag_exp, C) # m * nlabels
        self.grad_W = 1/m * np.matmul(X, e)

    def grad_inp(self, X, W, C):
        """
        :self.W : dim(L-1) * nlabels weights
        :param X: dim(L-1) * m (m number of examples, n dimension of each exmaple)
        :param W : dim(L-1) * nlabels
        :param C: m * nlables
        :return: dim(L-1) * nlabels
        """
        m = X.shape[-1]

        # use for all calculations
        w_x = np.exp(np.matmul(W.T, X)) # n * m
        stacked_x_w = np.array(w_x.sum(axis=0)) # 1 * m
        rep_w_x = npm.repmat(stacked_x_w, w_x.shape[0], 1) # n * m
        div_w_x = np.divide(w_x, rep_w_x) # n * m
        subc_w_x = np.subtract(div_w_x, C.T) # n * m
        return 1/m * np.matmul(W, subc_w_x) # d * m

    def __call__(self, X, W, C):
        """
        :param X: dim(L-1) * m (m number of examples, n dimension of each exmaple)
        :param W : dim(L-1) * nlabels
        :param C: m * nlables
        :return: scalar
        """
        m = X.shape[-1]

        x_w = np.exp(X.T @ W) # m * nlabels
        stacked_x_w = np.array(x_w.sum(axis=1)) # m * 1
        diag = np.linalg.inv(np.diag(stacked_x_w)) # m * m diag
        log_out = np.log(diag @ x_w) # m * nlabels
        c_log_out = np.trace(C.T @ log_out)

        return -1/m * c_log_out

    def step(self, opt, W):
        opt.step(self.grad_W, W)


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





#
# def grad_test():
#     d = np.random.rand(5, 3)
#     x = np.random.rand(5, 2)
#     c = np.random.rand(2, 3)
#     c[1,:] = 1 - c[0,:]
#     print(c)
#     w = np.random.rand(5, 3)
#     normalized_d = d / np.linalg.norm(d)
#     epsilon = np.geomspace(0.5, 0.5 ** 20, 20)
#     i = range(20)
#     fx = cross_entropy_loss_apply(x, w, c)                     # f(w)
#     softmax_grad_ret = cross_entropy_grad_w(x, w, c)   # grad(w)
#     no_grad, w_grad = [], []
#     for e in epsilon:
#         e_normalized_d = e * normalized_d           # e * d
#         w_perturbatzia = np.add(w, e_normalized_d)  # w + e*d
#         fx_d = cross_entropy_loss_apply(x, w_perturbatzia, c)  # f(w + e*d)
#         normalized_d_r = normalized_d.ravel()       # d
#         # print(np.linalg.norm(normalized_d_r))
#         softmax_grad_ret_r = softmax_grad_ret.ravel()
#         print('epsilon: ', e)
#         no_grad.append(abs(fx_d - fx))
#         w_grad.append(abs(fx_d - fx - e * normalized_d_r.T @ softmax_grad_ret_r))
#         print('fx_d - fx:', abs(fx_d - fx))
#         print('fx_d - fx - grad: ', abs(fx_d - fx - e * normalized_d_r.T @ softmax_grad_ret_r))
#
#     plt.plot(i, no_grad, 'k', label='No gradient')
#     plt.plot(i, w_grad, 'g', label='With gradient')
#     plt.yscale('log')
#     plt.legend()
#     plt.show()
#
#
# def new_grad_test():
#     x = np.random.rand(5, 10)  # 5 features, 10 samples
#
#     d = np.random.rand(5, 3)   # 5 features, 3 labels
#     d = d / np.linalg.norm(d)
#
#     w = np.random.rand(5, 3)   # 5 features, 3 labels
#
#     labels = np.random.randint(3, size=10)  # random draw of labels for 10 samples
#     c = np.zeros((labels.size, 3))  # 10 samples, 3 labels
#     c[np.arange(labels.size), labels] = 1  # columns in c are one-hot encoded
#
#     f_x = cross_entropy_loss_apply(x, w, c)  # compute f(x)
#     grad_x = cross_entropy_grad_w(x, w, c)  # compute grad(x)
#
#     no_grad, w_grad = [], []
#     eps_vals = np.geomspace(0.5, 0.5 ** 20, 20)
#     for eps in eps_vals:
#         eps_d = eps * d
#         f_xed = cross_entropy_loss_apply(x, w + eps_d, c)  # compute f(x + ed)
#
#         o_eps = abs(f_xed - f_x)
#         o_eps_sq = abs(f_xed - f_x - eps_d.ravel().T @ grad_x.ravel())
#         print(o_eps)
#         print(o_eps_sq)
#         no_grad.append(o_eps)
#         w_grad.append(o_eps_sq)
#
#     l = range(20)
#     plt.plot(l, no_grad, 'k', label='No gradient')
#     plt.plot(l, w_grad, 'g', label='With gradient')
#     plt.yscale('log')
#     plt.legend()
#     plt.show()


# def sgd_test():
#
#     Xtrain, Xtest, Ytrain, Ytest = loadGMMData()
#
#     learning_rate = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
#     batch_size = np.geomspace(2, 2 ** 8, 8)
#     Xtrain, Ytrain = shuffle(Xtrain, Ytrain, random_state=2021)
#
#     # split the data into train sets of different mini batch sizes
#     train_sets = [(np.array_split(Xtrain, Xtrain.shape[-1] / i, axis=1),
#                    np.array_split(Ytrain, Ytrain.shape[-1] / i, axis=1)) for i in batch_size]
#     test_sets = [(np.array_split(Xtest, Xtest.shape[-1] / i, axis=1),
#                   np.array_split(Ytest, Ytest.shape[-1] / i, axis=1)) for i in batch_size]
#
#     epochs = 7
#
#     W = np.random.randn(Ytrain.shape[0], 5) * np.sqrt(2 / 5)
#
#     # train loop
#     loss = 0
#     for i, bs in enumerate(batch_size):
#         all_batches, all_labels = train_sets[i]
#         for lr in learning_rate:
#
#             accs_hyper_params_train = []
#             accs_hyper_params_test = []
#
#             for e in range(epochs):
#                 acc_train = []
#                 loss_l = []
#                 for batch, labels in tqdm(zip(all_batches, all_labels)):
#
#                     W = sgd(cross_entropy_grad_w, batch, W, labels.T)
#
#                     loss = cross_entropy_loss_apply(batch, W, labels.T)
#                     loss_l.append(loss)
#
#
#                     # calculate train error
#                     labels = np.asarray([np.where(l == 1)[0][0] for l in labels.T])
#                     prediction = np.asarray([p[0] for p in softmax_predict(batch, W)])
#
#                     acc_train = np.append(acc_train, prediction == labels, axis=0)
#
#                 print('Epoch {} train acc: {}  train loss: {}'.format(e, np.mean(acc_train), np.mean(loss_l)))
#
#                 accs_hyper_params_train.append(np.mean(acc_train))
#                 accs_hyper_params_test.append(np.mean(test_accuracy(W, test_sets)))
#
#             plt.plot(range(epochs), accs_hyper_params_train, label='train error')
#             plt.plot(range(epochs), accs_hyper_params_test, label='validation error')
#             plt.title('Error of lr={} and batch size={}'.format(lr, bs))
#             plt.legend()
#             plt.show()


# def test_accuracy(W, test_sets):
#     # test loop
#     acc_test = []
#     all_batches, all_labels = test_sets[2]
#     for batch, labels in zip(all_batches, all_labels):
#         # calculate train error
#         # prediction = softmax_predict(batch, W)
#
#         labels = np.asarray([np.where(l == 1)[0][0] for l in labels.T])
#         prediction = np.asarray([p[0] for p in softmax_predict(batch, W)])
#
#         acc_test = np.append(acc_test, prediction == labels, axis=0)
#     print('Test acc: {}'.format(np.mean(acc_test)))
#     return acc_test


# sgd_test()
# new_grad_test()
# x = np.random.rand(5, 4)
# w = np.random.rand(5, 3)
# softmax_predict(x, w)
# sgd(softmax_grad, np.eye(4), np.random.rand(3, 6), np.eye(4), batch_size=2)
