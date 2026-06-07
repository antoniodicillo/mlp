import numpy as np
from .activations import relu, relu_derivative, softmax
from .losses import cross_entropy, one_hot_enconde

class MLP:
    def __init__(self, layer_sizes):
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            entrada = layer_sizes[i]
            saida   = layer_sizes[i + 1]

            W = np.random.randn(entrada, saida) * 0.01
            b = np.zeros((1, saida))

            self.weights.append(W)
            self.biases.append(b)
