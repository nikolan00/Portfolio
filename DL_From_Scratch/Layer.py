class Layer:

    def __init__(self):
        raise NotImplementedError()
    
    def forward(self, X):
        raise NotImplementedError()
    
    def backward(self, grad):
        raise NotImplementedError()