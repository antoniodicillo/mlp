import numpy as np

# Retorna o loss (grande = rede indo mal, pequeno = rede indo bem)
# Quando o loss vai dimuindo, a rede está aprendendo
def cross_entropy(y_true, y_pred):
    # Para evitar log(0)
    eps = 1e-8

    return -np.mean(
        np.sum(
            y_true * np.log(y_pred + eps),
            axis=1
        )
    )

# Simplifica a representação para 0 e 1
# 0 = classe incorreta, 1 = classe correta
def one_hot(y, num_classes=10):
    result = np.zeros((len(y), num_classes))
    result[np.arange(len(y)), y] = 1
    return result
