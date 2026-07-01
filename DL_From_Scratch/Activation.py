import numpy as np
from Layer import Layer

# 1 / (1 + e**-x)
class Sigmoid(Layer):
    def __init__(self):
        pass

    def forward(self, X):
        self.cache = X
        return 1. / (1. + np.e**(-X))

    def backward(self, _):
        ds = (np.e**(-self.cache)) / (1.+np.e**(-self.cache))**2.
        self.cache = None
        return ds
    

class ReLU(Layer):
    def __init__(self):
        pass

    def forward(self, X):
        X_ = np.where(X<=0, 0, X)
        self.cache = X_
        return X_

    def backward(self, _):
        dr = np.where(self.cache<=0, 0, 1)
        self.cache = None
        return dr