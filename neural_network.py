import numpy as np
from linear_layer import Linear
from activations import tanh
from functionals import Softmax, CrossEntropy


class Net:

    def __init__(self, n_layer, dim_in, dim_out, opt):

        self.act = tanh
        self.dim_L = 128
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_layer = n_layer
        self.layers = []
        self.hidden_units = []
        self.linear_inp = None
        self.opt = opt

        self.softmax = Softmax(self.dim_L if n_layer > 0 else self.dim_in, self.dim_out)
        self.cross_entropy = CrossEntropy()

    def _init_layers(self):

        self.linear_inp = Linear(self.dim_in, self.dim_L, self.act)
        for n in range(self.n_layer):
            self.layers.append(Linear(self.dim_L, self.dim_L, self.act))

    def __call__(self, input_, labels):
        """
        forward pass of the network
        :param input_:
        :param labels:
        :return:
        """

        # applies also the activation function
        self.hidden_units.append(self.linear_inp(input_))

        # forward pass through hidden layers
        for layer in self.layers:
            self.hidden_units.append(layer(self.hidden_units[-1]))

        return self.softmax(input_)

    def backward(self, input_, labels):

        backward_grad = []

        hidden_units = self.hidden_units
        hidden_units.reverse()

        # cross entropy grad
        self.cross_entropy.grad_w(hidden_units[0], self.softmax.W, labels)
        inp_grads = self.cross_entropy.grad_inp(hidden_units[0], self.softmax.W, labels)

        for i, layer in enumerate(self.layers, 1):
            layer.backward(hidden_units[i], inp_grads)
            inp_grads = layer.g_x

        self.linear_inp.backward(input_, inp_grads)

    def step(self):

        self.cross_entropy.step(self.opt, self.softmax.W)

        for layer in self.layers:
            layer.step(self.opt)

        self.linear_inp.step(self.opt)

# def softmax_predict(X, W):
#     softmax_res = softmax(X, W)
#     max_args = np.argmax(softmax_res, axis=1).reshape(-1, 1)
#     return max_args