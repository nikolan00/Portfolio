import numpy as np

class Model:
    
    def __init__(self, layers, lr=0.01):
        self.layers = layers
        self.trainable_layers = []
        self.lr = lr
        for layer in layers:
            if hasattr(layer, 'trainable'):
                self.trainable_layers.append(layer)


    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        
        return X

    def backward(self, grad):
        for layer in reversed(self.layers):
            dl = layer.backward(grad)
            if dl.shape == grad.shape:
                grad *= dl
            else:
                grad = np.dot(grad, dl.T)

    def update(self):
        for layer in self.trainable_layers:
            layer.update(self.lr)