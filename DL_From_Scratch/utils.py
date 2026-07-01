import numpy as np

def accuracy(y, y_p):
    y_p = np.array(y_p).reshape(y.shape)
    dist = np.abs(y-y_p)
    correct = np.where(dist<0.5, 1, 0)
    return np.sum(correct)*1. / len(y)

def normalize_data(X_train, X_val, X_test):
    smallest = np.min(X_train)
    X_train -= smallest
    largest = np.max(X_train)
    X_train /= largest / 2
    X_train -= 1

    X_val -= smallest
    X_val /= largest / 2
    X_val -= 1

    X_test -= smallest
    X_test /= largest / 2
    X_test -= 1

    return X_train, X_val, X_test