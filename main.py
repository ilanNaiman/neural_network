import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from scipy.sparse import coo_matrix
from data_loader import loadGMMData, loadSwissRollData, loadPeaksData
from functionals import *
from tqdm import tqdm
from optimizers import *
from neural_network import Net

parser = argparse.ArgumentParser(description='FeedForward Neural Network')

parser.add_argument('--iter', type=int, default=15,
                    help='number of iteration to train (default: 15)')
# parser.add_argument('--optimizer', type=str, default='SGD',
#                     choices=['SGD'], help='optimizer: choices SGD')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate (default: 0.1)')
parser.add_argument('--lr_decay', type=float, default=0.01, help='learning rate decay value (default: 0.00996)')
parser.add_argument('--lr_decay_epoch', type=int, nargs='+', default=[45],
                    help='decrease learning rate at these epochs.')
parser.add_argument('--lr_increase', type=float, default=3e-3, help='learning rate decay value (default: 3e-3)')
parser.add_argument('--lr_increase_epoch', type=int, nargs='+', default=[5],
                    help='decrease learning rate at these epochs.')
parser.add_argument('--batch_size', type=int, default=64,
                    help='size of batch size (default: 64)')
parser.add_argument('--n_layers', type=int, default=4,
                    help='number of layers (default: 4)')
parser.add_argument('--neurons', type=int, default=200,
                    help='size of neurons in each layer (default: 200)')
parser.add_argument('--data_set', type=str, default='PeaksData',
                    choices=['SwissRollData', 'GMMData', 'PeaksData'],
                    help='data set: choices SwissRoll, GMMData, PeaksData')

args = parser.parse_args()
print(args)


def main():
    # Load data
    if args.data_set == 'SwissRollData':
        Xtrain, Xtest, Ytrain, Ytest = loadSwissRollData()
    elif args.data_set == 'GMMData':
        Xtrain, Xtest, Ytrain, Ytest = loadGMMData()
    else:
        Xtrain, Xtest, Ytrain, Ytest = loadPeaksData()

    # Define set of learning rate and batch size (use only for testing)
    batch_size = np.geomspace(2, 2 ** 8, 8)
    batch_size = [round_(i) for i in batch_size]

    # preprocess data - shuffle and split into different batch sizes (using batch_size list)
    Xtrain, Ytrain, test_sets, train_sets = preprocess_data(Xtest, Xtrain, Ytest, Ytrain)

    # hyper params
    n_layer = args.n_layers
    neurons = args.neurons
    dim_in = Xtrain.shape[0]
    dim_out = Ytrain.shape[0]
    epochs = args.iter
    lr = args.lr
    opt = SGD(lr=lr)

    # init model
    model = Net(n_layer, dim_in, dim_out, opt, neurons)

    # train loop
    all_batches, all_labels = train_sets

    accs_hyper_params_train = []
    accs_hyper_params_test = []

    for e in range(1, epochs):
        acc_train = []
        loss_l = []
        for batch, labels in tqdm(zip(all_batches, all_labels), total=len(all_batches),
                                  file=sys.stdout):
            labels = labels.T

            outputs = model(batch, labels)

            loss = model.cross_entropy(outputs, labels)
            loss_l.append(loss)

            model.backward()
            model.step()

            outputs = model.softmax(outputs)

            # calculate train error
            labels = get_index(labels)
            prediction = predict(outputs)

            acc_train = np.append(acc_train, prediction == labels, axis=0)

        print('Epoch {} train acc: {}  train loss: {}'.format(e, np.mean(acc_train), np.mean(loss_l)))

        if e % args.lr_increase_epoch[0] == 0:
            lr += args.lr_increase
            model.opt = SGD(lr=lr)

        if e == args.lr_decay_epoch[0]:
            model.opt = SGD(lr=args.lr_decay)

        accs_hyper_params_train.append(np.mean(acc_train))
        accs_hyper_params_test.append(np.mean(test_accuracy(model, test_sets)))

    plt.plot(range(1, epochs), accs_hyper_params_train, label='Train Accuracy')
    plt.plot(range(1, epochs), accs_hyper_params_test, label='Validation Accuracy')
    plt.title('{} Data set, lr={:.5f} and batch size={}'.format(args.data_set, args.lr, args.batch_size))
    plt.legend()
    plt.savefig('./Test_Figures/{} Data set, lr={:.5f} and batch size={}.pdf'.format(args.data_set, args.lr,
                                                                                     args.batch_size),
                transparent=True, bbox_inches='tight', pad_inches=0)
    plt.savefig('./Test_Figures/{} Data set, lr={:.5f} and batch size={}.png'.format(args.data_set, args.lr,
                                                                                     args.batch_size),
                transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()


def test_accuracy(model, test_sets):
    # test loop
    acc_test = []
    loss_test = []
    all_batches, all_labels = test_sets

    for batch, labels in tqdm(zip(all_batches, all_labels), total=len(all_batches),
                              file=sys.stdout):
        # transpose the
        labels = labels.T

        # calculate test acc
        outputs = model(batch, labels)

        loss = model.cross_entropy(outputs, labels)
        loss_test.append(loss)

        outputs = model.softmax(outputs)

        # calculate train error
        labels = get_index(labels)
        prediction = predict(outputs)

        acc_test = np.append(acc_test, prediction == labels, axis=0)

    print('Test acc: {}'.format(np.mean(acc_test)))
    return acc_test


def preprocess_data(Xtest, Xtrain, Ytest, Ytrain):
    """

    :param Xtest: the validation examples set
    :param Xtrain: the train examples set
    :param Ytest: the validation labels
    :param Ytrain: the train labels
    :param batch_size: batch size
    :return:
    """

    nlabels = Ytrain.shape[0]
    Ytrain_shuffle = get_index_transpose(Ytrain)
    Xtrain_shuffle = Xtrain.T
    X_sparse = coo_matrix(Xtrain_shuffle)
    Xtrain, X_sparse, Ytrain = shuffle(Xtrain_shuffle, X_sparse, Ytrain_shuffle, random_state=2021)
    Ytrain = get_onehot(Ytrain, nlabels).T
    Xtrain = Xtrain.T

    # split the data into train sets of different mini batch sizes
    train_sets = prepare_batches(Xtrain, Ytrain)
    test_sets = prepare_batches(Xtest, Ytest)

    return Xtrain, Ytrain, test_sets, train_sets


def round_(x):
    return int(round(x))


def prepare_multiple_batches(X, Y, batch_size):
    return [(np.array_split(X, round_(X.shape[-1] / i), axis=1),
             np.array_split(Y, round_(Y.shape[-1] / i), axis=1)) for i in batch_size]


def prepare_batches(X, Y):
    return (np.array_split(X, round_(X.shape[-1] / args.batch_size), axis=1),
            np.array_split(Y, round_(Y.shape[-1] / args.batch_size), axis=1))


def init_weights(prev_layer, next_layer):
    return np.random.randn(prev_layer, next_layer) * np.sqrt(2 / next_layer)


def get_index(labels):
    return np.asarray([np.where(l == 1)[0][0] for l in labels])


def get_index_transpose(labels):
    return np.asarray([np.where(l == 1)[0][0] for l in labels.T])


def get_onehot(labels, nlables):
    c = np.zeros((labels.size, nlables))  # num of samples, num of labels
    c[np.arange(labels.size), labels] = 1  # columns in c are one-hot encoded
    return c


def predict(output):
    return np.asarray([p[0] for p in softmax_predict(output)])


def softmax_predict(output):
    max_args = np.argmax(output, axis=1).reshape(-1, 1)
    return max_args


if __name__ == '__main__':
    main()
