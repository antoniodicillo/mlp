import numpy as np

# Rectified Linear Unit, faz com que os valores negativos sejam convertidos para zero
def relu(x): 
    return np.maximum(0, x)

# Derivada do ReLu, 1 demonstra neuronios ativos, 0 demonstra neurônios inativos
def relu_derivative(x):
    return (x > 0).astype(float)

# Normaliza os valores para que a soma seja 1
def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp)


