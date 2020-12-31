import numpy as np
from linear_layer import Linear
from activations import tanh
from softmax import


class Net:

    def __init__(self, n_layer, dim_in, dim_out):

        self.act = tanh
        self.dim_L = 128
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_layer = n_layer
        self.layers = []
        self.linear_inp = None
        self.softmax = None

    def _init_layers(self):
        self.linear_inp = Linear(self.dim_in, self.dim_L, self.act)
        for n in range(self.n_layer):
            self.layers.append(Linear(self.dim_L, self.dim_L, self.act))





def softmax_predict(X, W):
    softmax_res = softmax(X, W)
    max_args = np.argmax(softmax_res, axis=1).reshape(-1, 1)
    return max_args