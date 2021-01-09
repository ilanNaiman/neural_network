import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from scipy.sparse import coo_matrix
from data_loader import loadGMMData, loadSwissRollData, loadPeaksData
from functionals import *
from tqdm import tqdm
from optimizers import *
from neural_network import Net


def main():
    # Load data

    Xtrain, Xtest, Ytrain, Ytest = loadSwissRollData()

    # Define set of learing rate and batch size
    learning_rate = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    batch_size = np.geomspace(2, 2 ** 8, 8)
    batch_size = [round_(i) for i in batch_size]

    nlabels = Ytrain.shape[0]
    Ytrain_shuffle = get_index2(Ytrain)
    Xtrain_shuufle = Xtrain.T
    X_sparse = coo_matrix(Xtrain_shuufle)
    Xtrain, X_sparse, Ytrain = shuffle(Xtrain_shuufle, X_sparse, Ytrain_shuffle, random_state=2021)
    Ytrain = get_onehot(Ytrain, nlabels).T
    Xtrain = Xtrain.T

    # split the data into train sets of different mini batch sizes
    train_sets = prepare_batches(Xtrain, Ytrain, batch_size)
    test_sets = prepare_batches(Xtest, Ytest, batch_size)

    # hyper params
    n_layer = 4
    dim_in = Xtrain.shape[0]
    dim_out = Ytrain.shape[0]
    epochs = 60
    lr = 0.00999
    opt = SGD(lr=lr)

    model = Net(n_layer, dim_in, dim_out, opt)

    # train loop

    all_batches, all_labels = train_sets[5]

    accs_hyper_params_train = []
    accs_hyper_params_test = []

    for e in range(1, epochs):
        acc_train = []
        loss_l = []
        for batch, labels in tqdm(zip(all_batches, all_labels)):

            labels = labels.T

            outputs = model(batch, labels)

            loss = model.cross_entropy(outputs, labels)
            loss_l.append(loss)

            model.backward()
            model.step()

            outputs = model.softmax(outputs)
            # calculate train error
            labels = get_index(labels)
            prediction = predict(model, outputs)

            acc_train = np.append(acc_train, prediction == labels, axis=0)

        print('Epoch {} train acc: {}  train loss: {}'.format(e, np.mean(acc_train), np.mean(loss_l)))

        if e % 10 == 0:
            lr += 3e-3
            model.opt = SGD(lr=lr)

        if e == 45:
            model.opt = SGD(lr=0.00996)

        accs_hyper_params_train.append(np.mean(acc_train))
        accs_hyper_params_test.append(np.mean(test_accuracy(model, test_sets[5])))


    plt.plot(range(1, epochs), accs_hyper_params_train, label='Train Accuracy')
    plt.plot(range(1, epochs), accs_hyper_params_test, label='Validation Accuracy')
    plt.title('Acc of lr={} and batch size={}'.format(lr, 64))
    plt.legend()
    plt.show()

    # plt.plot(range(epochs), accs_hyper_params_train, label='Train Accuracy')
    # plt.plot(range(epochs), accs_hyper_params_test, label='Validation Accuracy')
    # # plt.title('Acc of lr={} and batch size={}'.format(lr, bs))
    # plt.legend()
    # plt.show()


def test_accuracy(model, test_sets):
    # test loop
    acc_test = []
    loss_test = []
    all_batches, all_labels = test_sets
    for batch, labels in tqdm(zip(all_batches, all_labels)):
        labels = labels.T
        # calculate test acc
        outputs = model(batch, labels)

        loss = model.cross_entropy(outputs, labels)
        loss_test.append(loss)

        outputs = model.softmax(outputs)

        # calculate train error
        labels = get_index(labels)
        prediction = predict(model, outputs)

        acc_test = np.append(acc_test, prediction == labels, axis=0)

    print('Test acc: {}'.format(np.mean(acc_test)))
    return acc_test


def round_(x):
    return int(round(x))


def prepare_batches(X, Y, batch_size):
    return [(np.array_split(X, round_(X.shape[-1] / i), axis=1),
            np.array_split(Y, round_(Y.shape[-1] / i), axis=1)) for i in batch_size]


def init_weights(prev_layer, next_layer):
    return np.random.randn(prev_layer, next_layer) * np.sqrt(2 / next_layer)


def get_index(labels):
    return np.asarray([np.where(l == 1)[0][0] for l in labels])


def get_index2(labels):
    out = []
    for l in labels.T:
        jj = np.where(l == 1)
        out.append(jj[0][0])
    return np.asarray(out)


def get_onehot(labels, nlables):
    c = np.zeros((labels.size, nlables))  # 10 samples, 3 labels
    c[np.arange(labels.size), labels] = 1  # columns in c are one-hot encoded
    return c


def predict(model, output):
    return np.asarray([p[0] for p in softmax_predict(model, output)])


def softmax_predict(model, output):
    max_args = np.argmax(output, axis=1).reshape(-1, 1)
    return max_args


if __name__ == '__main__':
    main()
