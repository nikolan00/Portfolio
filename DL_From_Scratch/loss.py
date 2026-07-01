# -y * ln(y') - (1-y)*ln(1-y')
import numpy as np

class BCE:
      
    def forward(y, y_p):
        y_p = y_p.reshape(y.shape)
        loss = - y * np.log(y_p) - (1-y) * np.log(1-y_p)
        return loss

    def backward(y, y_p):
        y = y.reshape(y_p.shape)
        dl = - y / y_p + (1-y) / (1-y_p)
        return dl