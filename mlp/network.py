import numpy as np
from .activations import relu, relu_derivative, softmax
from .losses import cross_entropy

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
    # Embaralha os dados e divide em batches
    def fit(self, X, y, optimizer, epochs=10, batch_size=64, X_val=None, y_val=None):
        self.history_loss = []
        self.history_acc = []
        self.history_val_loss = []
        self.history_val_acc = []

        for epoch in range(epochs):
            # Embaralha os dados
            indexes = np.random.permutation(len(X))
            X, y = X[indexes], y[indexes]

            total_loss = 0

            for begin in range(0, len(X), batch_size):
                X_batch = X[begin:begin + batch_size]
                y_batch = y[begin:begin + batch_size]

                probs = self.forward(X_batch)
                loss = cross_entropy(y_batch, probs)
                self.backward(y_batch)
                optimizer.step(self)

                total_loss += loss

            loss_media = total_loss / (len(X) // batch_size)
            acc = self.accuracy(X, y)

            self.history_loss.append(loss_media)
            self.history_acc.append(acc)

            log = f"Epoch {epoch + 1}/{epochs} - Loss: {loss_media:.4f} - Acc: {acc:.4f}"

            if X_val is not None and y_val is not None:
                val_probs = self.forward(X_val)
                val_loss = cross_entropy(y_val, val_probs)
                val_acc = self.accuracy(X_val, y_val)
                self.history_val_loss.append(val_loss)
                self.history_val_acc.append(val_acc)
                log += f" - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}"

            print(log)

    # Compara a classe com a maior probabilidade com o rótulo real
    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        y_true = np.argmax(y, axis=1)
        return np.mean(y_pred == y_true)