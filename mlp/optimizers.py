# Utiliza os gradientes calculados para atualizar os pesos e bias
class SGD:    
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def step(self, network):
        for i in range(len(network.weights)):
            network.weights[i] -= self.lr * network.dWs[i]
            network.biases[i] -= self.lr * network.dbs[i]
            