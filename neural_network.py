import numpy as np
from linear_layer import Linear
from activations import tanh
from functionals import Softmax, CrossEntropy


class Net:

    def __init__(self, n_layer, dim_in, dim_out, opt):

        assert n_layer > 0

        self.act = tanh
        self.dim_L = 200
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_layer = n_layer
        self.layers = []
        self.hidden_units = []
        self.linear_inp = None
        self.opt = opt
        self.labels = None

        self.softmax = None
        self.cross_entropy = None

        self._init_layers()

    def _init_layers(self):

        if self.n_layer == 1:
            self.softmax = Softmax(self.dim_in, self.dim_out)

        else:
            self.linear_inp = Linear(self.dim_in, self.dim_L, self.act)
            for n in range(self.n_layer - 2):
                self.layers.append(Linear(self.dim_L, self.dim_L, self.act))
            self.softmax = Softmax(self.dim_L, self.dim_out)
            self.cross_entropy = CrossEntropy(self.softmax.W)

    def __call__(self, input_, labels):
        """
        forward pass of the network
        :param input_:
        :param labels:
        :return:
        """

        self.hidden_units = [input_]
        # save labels for backward pass
        self.labels = labels

        if self.linear_inp is not None:
            # applies also the activation function
            self.hidden_units.append(self.linear_inp(input_))

        # forward pass through hidden layers
        for layer in self.layers:
            self.hidden_units.append(layer(self.hidden_units[-1]))

        return self.hidden_units[-1]

    def backward(self):

        hidden_units = self.hidden_units
        hidden_units.reverse()

        # cross entropy grad
        self.cross_entropy.grad_w(hidden_units[0], self.labels)
        inp_grads = self.cross_entropy.grad_inp(hidden_units[0], self.labels)

        # linear layers grads in reverse order
        for i, layer in enumerate(reversed(self.layers), 1):
            layer.backward(hidden_units[i], inp_grads)
            inp_grads = layer.g_x

        if self.linear_inp is not None:
            self.linear_inp.backward(hidden_units[-1], inp_grads)

    def step(self):

        self.cross_entropy.step(self.opt)
        self.softmax.W = self.cross_entropy.W

        for layer in reversed(self.layers):
            layer.step(self.opt)

        if self.linear_inp is not None:
            self.linear_inp.step(self.opt)
