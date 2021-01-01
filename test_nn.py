import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from data_loader import loadGMMData, loadSwissRollData
from functionals import *
from optimizers import SGD
from tqdm import tqdm


def round_(x):
    return int(round(x))


def prepare_batches(X, Y, batch_size):
    return [(np.array_split(X, round_(X.shape[-1] / i), axis=1),
            np.array_split(Y, round_(Y.shape[-1] / i), axis=1)) for i in batch_size]


def init_weights(prev_layer, next_layer):
    return np.random.randn(prev_layer.shape[0], next_layer.shape[0]) * np.sqrt(2 / 5)


def get_index(labels):
    return np.asarray([np.where(l == 1)[0][0] for l in labels])


def predict(output):
    return np.asarray([p[0] for p in softmax_predict(output)])


def softmax_predict(output):
    max_args = np.argmax(output, axis=1).reshape(-1, 1)
    return max_args


def new_grad_test():
    x = np.random.rand(5, 10)  # 5 features, 10 samples

    d = np.random.rand(5, 3)   # 5 features, 3 labels
    d = d / np.linalg.norm(d)

    w = np.random.rand(5, 3)   # 5 features, 3 labels

    labels = np.random.randint(3, size=10)  # random draw of labels for 10 samples
    c = np.zeros((labels.size, 3))  # 10 samples, 3 labels
    c[np.arange(labels.size), labels] = 1  # columns in c are one-hot encoded

    loss_func = CrossEntropy()

    f_x = loss_func(x, w, c)  # compute f(x)
    loss_func.grad_w(x, w, c)  # compute grad(x)
    grad_x = loss_func.grad_W

    no_grad, w_grad = [], []
    eps_vals = np.geomspace(0.5, 0.5 ** 20, 20)
    for eps in eps_vals:
        eps_d = eps * d
        f_xed = loss_func(x, w + eps_d, c)  # compute f(x + ed)

        o_eps = abs(f_xed - f_x)
        o_eps_sq = abs(f_xed - f_x - eps_d.ravel().T @ grad_x.ravel())
        print(o_eps)
        print(o_eps_sq)
        no_grad.append(o_eps)
        w_grad.append(o_eps_sq)

    l = range(20)
    plt.plot(l, no_grad, 'k', label='No gradient')
    plt.plot(l, w_grad, 'g', label='With gradient')
    plt.yscale('log')
    plt.legend()
    plt.show()


def sgd_test():

    # Load data
    Xtrain, Xtest, Ytrain, Ytest = loadSwissRollData()

    # Define set of learing rate and batch size
    learning_rate = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    batch_size = np.geomspace(2, 2 ** 8, 8)
    batch_size = [round_(i) for i in batch_size]

    Xtrain, Ytrain = shuffle(Xtrain, Ytrain, random_state=2021)

    # split the data into train sets of different mini batch sizes
    train_sets = prepare_batches(Xtrain, Ytrain, batch_size)
    test_sets = prepare_batches(Xtest, Ytest, batch_size)

    epochs = 7

    # train loop
    for i, bs in enumerate(batch_size):

        all_batches, all_labels = train_sets[i]
        for lr in learning_rate:

            loss_func = CrossEntropy()
            opt = SGD(lr=lr)
            softmax = Softmax(Xtrain.shape[0], Ytrain.shape[0])

            accs_hyper_params_train = []
            accs_hyper_params_test = []

            for e in range(epochs):
                acc_train = []
                loss_l = []
                for batch, labels in tqdm(zip(all_batches, all_labels)):

                    labels = labels.T

                    loss_func.grad_w(batch, softmax.W, labels)
                    softmax.W = opt.step(loss_func.grad_W, softmax.W)

                    loss = loss_func(batch, softmax.W, labels)
                    loss_l.append(loss)

                    output = softmax(batch)
                    # calculate train error
                    labels = get_index(labels)
                    prediction = predict(output)

                    acc_train = np.append(acc_train, prediction == labels, axis=0)

                print('Epoch {} train acc: {}  train loss: {}'.format(e, np.mean(acc_train), np.mean(loss_l)))

                accs_hyper_params_train.append(np.mean(acc_train))
                accs_hyper_params_test.append(np.mean(test_accuracy(softmax, test_sets[i])))

            plt.plot(range(epochs), accs_hyper_params_train, label='Train Accuracy')
            plt.plot(range(epochs), accs_hyper_params_test, label='Validation Accuracy')
            plt.title('Acc of lr={} and batch size={}'.format(lr, bs))
            plt.legend()
            plt.show()


def test_accuracy(softmax, test_sets):
    # test loop
    acc_test = []
    all_batches, all_labels = test_sets
    for batch, labels in tqdm(zip(all_batches, all_labels)):

        # calculate test acc
        labels = get_index(labels.T)
        output = softmax(batch)
        prediction = predict(output)

        acc_test = np.append(acc_test, prediction == labels, axis=0)

    print('Test acc: {}'.format(np.mean(acc_test)))
    return acc_test


sgd_test()
