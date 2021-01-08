import numpy as np
import argparse
from utils import *


class BinaryLogisticRegression:
    def __init__(self, D):
        """
        D - number of features
        """
        self.D = D
        self.weights = np.random.rand(D, 1)

    def predict(self, X):
        """
        X - numpy array of shape (N, D)
        """
        # TODO: Return a (N, 1) numpy array of predictions.
        x = np.matmul(X,self.weights)
        probability =  1/(1+np.exp(-1*x))
        prediction = np.where(probability > 0.5,1,0)
        return prediction
        # END TODO

    def train(self, X, Y, lr=1, max_iter=2000):
        for _ in range(max_iter):
            # TODO: Update the weights using a single step of gradient descent. You are not allowed to use loops here.
            gradient = (1/len(Y)) * (np.matmul(X.T,1/(1+np.exp(-1*np.matmul(X,self.weights)))-Y))
            self.weights = self.weights - lr * gradient
            # END TODO
            # TODO: Stop the algorithm if the norm of the gradient falls below 1e-4
            if np.sqrt(np.sum(gradient**2)) <= 1e-4:
                break
            # End TODO

    def accuracy(self, preds, Y):
        """
        preds - numpy array of shape (N, 1) corresponding to predicted labels
        Y - numpy array of shape (N, 1) corresponding to true labels
        """
        accuracy = ((preds == Y).sum()) / len(preds)
        return accuracy

    def f1_score(self, preds, Y):
        """
        preds - numpy array of shape (N, 1) corresponding to predicted labels
        Y - numpy array of shape (N, 1) corresponding to true labels
        """
        # TODO: calculate F1 score for predictions preds and true labels Y
        TP = len([x for x,y in zip(list(preds),list(Y)) if y == 1 and x == 1])
        FP = len([x for x,y in zip(list(preds),list(Y)) if y == 0 and x == 1])
        FN = len([x for x,y in zip(list(preds),list(Y)) if y == 1 and x == 0])
        return 2*TP / (2*TP + FP + FN)
        # End TODO


if __name__ == '__main__':
    np.random.seed(335)

    X, Y = load_data('data/songs.csv')
    print(X.shape)
    X, Y = preprocess(X, Y)
    print(X.shape)
    X_train, Y_train, X_test, Y_test = split_data(X, Y)
    

    D = X_train.shape[1]

    lr = BinaryLogisticRegression(D)
    lr.train(X_train, Y_train)
    preds = lr.predict(X_test)
    acc = lr.accuracy(preds, Y_test)
    f1 = lr.f1_score(preds, Y_test)
    print('Test Accuracy: {}'.format(acc))
    print('Test F1 Score: {}'.format(f1))
