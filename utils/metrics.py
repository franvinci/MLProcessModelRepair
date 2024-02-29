import numpy as np

def Accuracy(y_true, y_pred):

    y_pred = y_pred.ravel()
    y_pred = (y_pred > 0.5)*1
    acc = np.mean(y_pred == y_true)

    return acc