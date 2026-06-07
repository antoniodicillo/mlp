import numpy as np
from .activations import relu, relu_derivative, softmax
from .losses import cross_entropy, one_hot_enconde

class MLP:
    def __init__(self, layer_sizes):
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            input = layer_sizes[i]
            output = layer_sizes[i + 1]

            W = np.random.randn(input, output) * 0.01
            b = np.zeros((1, output))  

            self.weights.append(W)
            self.biases.append(b) 
    # Pega a entrada X e passa por cada camada da rede
    def forward(self, X):
        self.Zs = []
        self.As = [X]

        A = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            Z = A @ W + b
            self.Zs.append(Z)

            last_layer = (i == len(self.weights) - 1)
            if(last_layer):
                A = softmax(Z)
            else:
                A = relu(Z)

            self.As.append(A)

        return A
    # Quanto cada peso e bias contribui para o erro
    def backward(self, y_true):
        self.dWs = []
        self.dbs = []
        batch_size = y_true.shape[0]

        dZ = self.As[-1] - y_true
        for i in reversed(range(len(self.weights))):
            A_old = self.As[i]

            dW = A_old.T @ dZ / batch_size
            db = np.sum(dZ, axis=0, keepdims=True) / batch_size

            self.dWs.insert(0, dW)
            self.dbs.insert(0, db)

            if i > 0:
                dA = dZ @ self.weights[i].T
                dZ = dA * relu_derivative(self.Zs[i - 1])

