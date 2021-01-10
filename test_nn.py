import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from data_loader import loadGMMData, loadSwissRollData, loadPeaksData
from functionals import *
from optimizers import SGD
from tqdm import tqdm
from linear_layer import Linear
from activations import tanh
import sys
import argparse
from scipy.sparse import coo_matrix


parser = argparse.ArgumentParser(description='Test FeedForward Neural Network')

parser.add_argument('--iter', type=int, default=5,
                    help='number of iteration to train (default: 60)')
# parser.add_argument('--optimizer', type=str, default='SGD',
#                     choices=['SGD'], help='optimizer: choices SGD')
parser.add_argument('--lr', type=float, default=0.005,
                    help='learning rate (default: 0.00999)')
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
parser.add_argument('--data_set', type=str, default='SwissRollData',
                    choices=['SwissRollData', 'GMMData', 'PeaksData'],
                    help='data set: choices SwissRoll, GMMData, PeaksData')
parser.add_argument('--jacMV_test', type=str, default='weights',
                    choices=['input', 'weights', 'bias'],
                    help='jacMV_test: choices input, weights, bias')
parser.add_argument('--jacTMV_test', type=str, default='weights',
                    choices=['input', 'weights', 'bias'],
                    help='jacTMV_test: choices input, weights, bias')

args = parser.parse_args()
print(args)


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


def new_grad_test():
    x = np.random.rand(5, 10)  # 5 features, 10 samples

    d = np.random.rand(5, 3)   # 5 features, 3 labels
    d = d / np.linalg.norm(d)

    w = np.random.rand(5, 3)   # 5 features, 3 labels

    labels = np.random.randint(3, size=10)  # random draw of labels for 10 samples
    c = np.zeros((labels.size, 3))  # 10 samples, 3 labels
    c[np.arange(labels.size), labels] = 1  # columns in c are one-hot encoded

    loss_func = CrossEntropy(w)

    f_x = loss_func(x, c)  # compute f(x)
    loss_func.grad_w(x, c)  # compute grad(x)
    grad_x = loss_func.grad_W
    w = loss_func.W

    no_grad, w_grad = [], []
    eps_vals = np.geomspace(0.5, 0.5 ** 20, 20)
    for eps in eps_vals:
        eps_d = eps * d
        loss_func.W = w + eps_d
        f_xed = loss_func(x, c)  # compute f(x + ed)

        o_eps = abs(f_xed - f_x)
        o_eps_sq = abs(f_xed - f_x - eps_d.ravel().T @ grad_x.ravel())
        print(o_eps)
        print(o_eps_sq)
        no_grad.append(o_eps)
        w_grad.append(o_eps_sq)

    l = range(20)
    plt.title('Softmax gradient test')
    plt.plot(l, no_grad, label='First Order')
    plt.plot(l, w_grad, label='Second Order')
    plt.yscale('log')
    plt.legend()
    plt.savefig('./Test_Figures/grad_test.png', transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()


def sgd_test():

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

    # train loop

    all_batches, all_labels = train_sets


    softmax = Softmax(Xtrain.shape[0] + 1, Ytrain.shape[0])
    loss_func = CrossEntropy(softmax.W)
    opt = SGD(lr=args.lr)


    accs_hyper_params_train = []
    accs_hyper_params_test = []

    for e in range(args.iter):
        acc_train = []
        loss_l = []
        for batch, labels in tqdm(zip(all_batches, all_labels), total=len(all_batches),
                                  file=sys.stdout):
            labels = labels.T

            ones = np.ones((1, batch.shape[-1]), dtype=int)
            batch = np.concatenate((batch, ones), axis=0)

            loss = loss_func(batch, labels)
            loss_l.append(loss)

            loss_func.grad_w(batch, labels)
            softmax.W = opt.step(loss_func.grad_W, softmax.W)
            loss_func.W = softmax.W

            output = softmax(batch)
            # calculate train error
            labels = get_index(labels)
            prediction = predict(output)

            acc_train = np.append(acc_train, prediction == labels, axis=0)

        print('Epoch {} train acc: {}  train loss: {}'.format(e, np.mean(acc_train), np.mean(loss_l)))

        accs_hyper_params_train.append(np.mean(acc_train))
        accs_hyper_params_test.append(np.mean(test_accuracy(softmax, test_sets)))

    plt.plot(range(args.iter), accs_hyper_params_train, label='Train Accuracy')
    plt.plot(range(args.iter), accs_hyper_params_test, label='Validation Accuracy')
    plt.title('{} Set, Acc of lr={} and batch size={}'.format(args.data_set, args.lr, args.batch_size))
    plt.legend()
    plt.savefig('./Test_Figures/{} Set, Acc of lr={} and batch size={}.png'
                .format(args.data_set, args.lr, args.batch_size),
                transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()


def test_accuracy(softmax, test_sets):
    # test loop
    acc_test = []
    all_batches, all_labels = test_sets
    for batch, labels in tqdm(zip(all_batches, all_labels), total=len(all_batches),
                              file=sys.stdout):

        ones = np.ones((1, batch.shape[-1]), dtype=int)
        batch = np.concatenate((batch, ones), axis=0)

        # calculate test acc
        labels = get_index(labels.T)
        output = softmax(batch)
        prediction = predict(output)

        acc_test = np.append(acc_test, prediction == labels, axis=0)

    print('Test acc: {}'.format(np.mean(acc_test)))
    return acc_test


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
        jackMV_x = lin1.jacMV_x(x, e_normalized_d)
        print(jackMV_x)
        print('epsilon: ', eps)
        print(np.linalg.norm(np.subtract(fx_d, fx)))
        print(np.linalg.norm(np.subtract(np.subtract(fx_d, fx), jackMV_x)))
        no_grad.append(np.linalg.norm(np.subtract(fx_d, fx)))
        x_grad.append(np.linalg.norm(np.subtract(np.subtract(fx_d, fx), jackMV_x)))

    l = range(eps_num)
    plt.plot(l, no_grad, label='First Order')
    plt.plot(l, x_grad, label='Second Order')
    plt.yscale('log')
    plt.legend()
    plt.title('jacMV test w.r.t {}'.format(args.jacMV_test))
    plt.savefig('./Test_Figures/jacMV test w.r.t {}.png'
                .format(args.jacMV_test),
                transparent=True, bbox_inches='tight', pad_inches=0)
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

        jackMV_b = lin1.jacMV_b(x, eps_d)

        no_grad.append(np.linalg.norm(np.subtract(fx_d, fx)))
        b_grad.append(np.linalg.norm(np.subtract(np.subtract(fx_d, fx), jackMV_b)))

    l = range(eps_num)
    plt.plot(l, no_grad, label='First Order')
    plt.plot(l, b_grad, label='Second Order')
    plt.yscale('log')
    plt.legend()
    plt.title('jacMV test w.r.t {}'.format(args.jacMV_test))
    plt.savefig('./Test_Figures/jacMV test w.r.t {}.png'
                .format(args.jacMV_test),
                transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()


def jacMV_w_test():
    d = np.random.rand(4, 3)
    x = np.random.rand(3, 1)
    # normalized_d = d / np.linalg.norm(d)

    eps_num = 20
    eps_vals = np.geomspace(0.5, 0.5 ** eps_num, eps_num)

    lin1 = Linear(3, 4, tanh)
    fx = lin1(x)

    no_grad, w_grad = [], []
    w = lin1.W

    for eps in eps_vals:

        eps_d = eps * d

        lin1.W = np.add(w, eps_d)
        fx_d = lin1(x)

        # eps_d = eps_d.reshape(-1, 1)
        jacMV_w = lin1.jacMV_w(x, eps_d)

        first_order = fx_d - fx
        second_order = first_order - jacMV_w

        no_grad.append(np.linalg.norm(first_order))
        w_grad.append(np.linalg.norm(second_order))

    l = range(eps_num)
    plt.plot(l, no_grad, label='First Order')
    plt.plot(l, w_grad, label='Second Order')
    plt.yscale('log')
    plt.legend()
    plt.title('jacMV test w.r.t {}'.format(args.jacMV_test))
    plt.savefig('./Test_Figures/jacMV test w.r.t {}.png'
                .format(args.jacMV_test),
                transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()


def jacTMV_w_test():
    v = np.random.rand(4, 3)
    x = np.random.rand(3, 1)
    u = np.random.rand(4, 1)
    lin1 = Linear(3, 4, tanh)

    jacMV_w = lin1.jacMV_w(x, v)

    lin1.backward(x, u)
    jacTMV_w = lin1.g_w

    u_jac = u.T @ jacMV_w
    v_jacT = v.ravel().T @ jacTMV_w.ravel()

    print(abs(np.subtract(u_jac, v_jacT)))


def jacTMV_b_test():
    v = np.random.rand(3, 1)
    x = np.random.rand(3, 1)
    u = np.random.rand(3, 1)
    lin1 = Linear(3, 3, tanh)

    jackMV_b = lin1.jacMV_b(x, v)


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

    jacMV_x = lin1.jacMV_x(x, v)


    lin1.backward(x, u)
    jacTMV_x = lin1.g_x


    u_jac = u.T @ jacMV_x
    v_jacT = v.T @ jacTMV_x

    print(abs(np.subtract(u_jac, v_jacT)))


def jacMV_test():

    if args.jacMV_test == 'input':
        jacMV_x_test()
    if args.jacMV_test == 'weights':
        jacMV_w_test()
    if args.jacMV_test == 'bias':
        jacMV_b_test()

    if args.jacTMV_test == 'input':
        jacTMV_x_test()
    if args.jacTMV_test == 'weights':
        jacTMV_w_test()
    if args.jacTMV_test == 'bias':
        jacTMV_b_test()


# section 6, gradient test for the whole network


# sgd_test()
jacMV_test()
# new_grad_test()
