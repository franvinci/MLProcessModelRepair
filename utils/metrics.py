import numpy as np


def SimilarityMetric(y_true, y_pred):

    y_pred = y_pred.ravel()
    y_pred = (y_pred > 0.5)*1
    acc = np.mean(y_pred == y_true)
    similarity = 1 - np.abs(acc - 0.5)
    similarity_scaled = (similarity - 0.5)/0.5

    return similarity_scaled


def Accuracy(y_true, y_pred):

    y_pred = y_pred.ravel()
    y_pred = (y_pred > 0.5)*1
    acc = np.mean(y_pred == y_true)

    return acc