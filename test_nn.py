import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from data_loader import loadGMMData
from functionals import *
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


def predict(batch, W):
    return np.asarray([p[0] for p in softmax_predict(batch, W)])


def sgd_test():

    # Load data
    Xtrain, Xtest, Ytrain, Ytest = loadGMMData()

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

            W = init_weights(Xtrain, Ytrain)

            accs_hyper_params_train = []
            accs_hyper_params_test = []

            for e in range(epochs):
                acc_train = []
                loss_l = []
                for batch, labels in tqdm(zip(all_batches, all_labels)):

                    labels = labels.T

                    W = sgd(cross_entropy_grad_w, batch, W, labels, lr=lr)

                    loss = cross_entropy_loss_apply(batch, W, labels)
                    loss_l.append(loss)

                    # calculate train error
                    labels = get_index(labels)
                    prediction = predict(batch, W)

                    acc_train = np.append(acc_train, prediction == labels, axis=0)

                print('Epoch {} train acc: {}  train loss: {}'.format(e, np.mean(acc_train), np.mean(loss_l)))

                accs_hyper_params_train.append(np.mean(acc_train))
                accs_hyper_params_test.append(np.mean(test_accuracy(W, test_sets[i])))

            plt.plot(range(epochs), accs_hyper_params_train, label='Train Accuracy')
            plt.plot(range(epochs), accs_hyper_params_test, label='Validation Accuracy')
            plt.title('Acc of lr={} and batch size={}'.format(lr, bs))
            plt.legend()
            plt.show()


def test_accuracy(W, test_sets):
    # test loop
    acc_test = []
    all_batches, all_labels = test_sets
    for batch, labels in tqdm(zip(all_batches, all_labels)):

        # calculate test acc
        labels = get_index(labels.T)
        prediction = predict(batch, W)

        acc_test = np.append(acc_test, prediction == labels, axis=0)

    print('Test acc: {}'.format(np.mean(acc_test)))
    return acc_test


sgd_test()
