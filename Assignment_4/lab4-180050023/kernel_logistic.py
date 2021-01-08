import numpy as np
from kernel import *
from utils import *
import matplotlib.pyplot as plt


class KernelLogistic(object):
    def __init__(self, kernel=gaussian_kernel, iterations=100,eta=0.01,lamda=0.05,sigma=1):
        self.kernel = lambda x,y: kernel(x,y,sigma)
        self.iterations = iterations
        self.alpha = None
        self.eta = eta     # Step size for gradient descent
        self.lamda = lamda # Regularization term

    def fit(self, X, y):
        ''' find the alpha values here'''
        self.train_X = X
        self.train_y = y
        self.alpha = np.zeros((y.shape[0],1))
        kernel = self.kernel(self.train_X,self.train_X)#gram matrix

        # TODO
        for idx in range(self.iterations):# K@(prediction - y + lamda*alpha)
            gradient = np.matmul(kernel,1.0/(1+np.exp(-np.matmul(kernel,self.alpha))) - self.train_y[:,None] + self.lamda * self.alpha)
            self.alpha = self.alpha - self.eta * gradient
        # END TODO
    

    def predict(self, X):
        # TODO
        kernel = self.kernel(X,self.train_X)#gram matrix
        return (1/(1+np.exp(-np.matmul(kernel,self.alpha)))).reshape(X.shape[0])
        # END TODO

def k_fold_cv(X,y,k=10,sigma=1.0):
    '''Does k-fold cross validation given train set (X, y)
    Divide train set into k subsets, and train on (k-1) while testing on 1. 
    Do this process k times.
    Do Not randomize 
    
    Arguments:
        X  -- Train set
        y  -- Train set labels
    
    Keyword Arguments:
        k {number} -- k for the evaluation
        sigma {number} -- parameter for gaussian kernel
    
    Returns:
        error -- (sum of total mistakes for each fold)/(k)
    '''
    # TODO 
    N = X.shape[0]
    fold_size = int(N/k)
    avg_error = 0
    for i in range(k):
        xtest = X[fold_size * i:fold_size * (i+1),:]
        ytest = y[fold_size * i:fold_size * (i+1)]
        xtrain = np.concatenate([X[:fold_size * i,:],X[fold_size * (i+1):,:]],axis=0)
        ytrain = np.concatenate([y[:fold_size * i],y[fold_size * (i+1):]],axis=0)
        clf = KernelLogistic(gaussian_kernel,sigma=sigma)
        clf.fit(xtrain,ytrain)
        y_predict = clf.predict(xtest) > 0.5
        avg_error += np.sum( y_predict != ytest)
    return avg_error/k
    # END TODO

if __name__ == '__main__':
    data = np.loadtxt("./data/dataset1.txt")
    X1 = data[:900,:2]
    Y1 = data[:900,2]

    clf = KernelLogistic(gaussian_kernel,sigma=1)
    clf.fit(X1, Y1)

    y_predict = clf.predict(data[900:,:2]) > 0.5

    correct = np.sum(y_predict == data[900:,2])
    print("%d out of %d predictions correct" % (correct, len(y_predict)))
    if correct > 92:
        marks = 1.0
    else:
        marks = 0
    print(f"You recieve {marks} for the fit function")

    errs = []
    sigmas = [0.5, 1, 2, 3, 4, 5, 6]
    for s in sigmas:  
      errs+=[(k_fold_cv(X1,Y1,sigma=s))]
    plt.plot(sigmas,errs)
    plt.xlabel('Sigma')
    plt.ylabel('Mistakes')
    plt.title('A plot of sigma v/s mistakes')
    plt.show()
