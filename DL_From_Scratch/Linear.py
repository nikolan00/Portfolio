# y = b +  Wx
import numpy as np
from Layer import Layer

class LinearLayer(Layer):

    def __init__(self, dim1, dim2, nonlin="relu"):
        np.random.seed(0)
        if nonlin == "relu":
            std = np.sqrt(2.0 / dim1)
            self.W = np.random.normal(0.0, std, size=(dim1, dim2))
        else:
            limit = np.sqrt(6 / (dim1 + dim2))
            self.W = np.random.uniform(-limit, limit, size=(dim1,dim2))
        self.b = np.zeros((dim2,1))
        self.trainable = [self.W, self.b]

    def forward(self, X):
        self.bs = X.shape[0]
        y = np.dot(X, self.W) + self.b.T
        self.cache = X
        return y

    def backward(self, grad):
        # grad: (dim2,bs)
        bs = self.cache.shape[0]
        dW = self.cache
        self.dW = np.dot(dW.T, grad)/bs
        self.db = np.dot(grad.T, np.ones((bs, 1)))/bs
        self.cache = None
        return self.W
    
    def update(self,lr):
        self.W = self.W - lr * self.dW
        self.b = self.b - lr * self.db