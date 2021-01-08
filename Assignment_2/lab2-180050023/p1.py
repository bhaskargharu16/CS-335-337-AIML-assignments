import numpy as np
import matplotlib.pyplot as plt
from utils import load_data2, split_data, preprocess, normalize
np.random.seed(337)


def mse(X, Y, W):
    """
    Compute mean squared error between predictions and true y values

    Args:
    X - numpy array of shape (n_samples, n_features)
    Y - numpy array of shape (n_samples, 1)
    W - numpy array of shape (n_features, 1)
    """

    # TODO
    mse = float(np.mean((np.matmul(X,W) - Y)**2,axis=0)/2)
    # END TODO

    return mse

def ridge_regression(X_train, Y_train, X_test, Y_test, reg, lr=0.001, max_iter=200):
	'''
	reg - regularization parameter (lambda in Q2.1 c)
	'''
	train_mses = []
	test_mses = []

	## TODO: Initialize W using using random normal 
	W = np.random.normal(size=(X_train.shape[1],1))*0.001
	## END TODO

	for i in range(max_iter):

		## TODO: Compute train and test MSE
		train_mse = mse(X_train,Y_train,W) 
		test_mse = mse(X_test,Y_test,W) 
		## END TODO

		train_mses.append(train_mse)
		test_mses.append(test_mse)

		## TODO: Update w and b using a single step of gradient descent
		diff = np.matmul(X_train,W) - Y_train
		gradient = (1/X_train.shape[0]) * np.matmul(np.transpose(X_train),diff) + 2*reg*W
		W = W - lr*gradient
		## END TODO

	return W, train_mses, test_mses

def ista(X_train, Y_train, X_test, Y_test, _lambda=0.1, lr=0.005, max_iter=10000):
    """
    Iterative Soft-thresholding Algorithm for LASSO
    """
    train_mses = []
    test_mses = []

    # TODO: Initialize W using using random normal
    W = np.random.normal(0,1,size=(X_train.shape[1],1))
    # END TODO
    # while True:
    for i in range(max_iter):
        # TODO: Compute train and test MSE
        train_mse = mse(X_train,Y_train,W)
        test_mse = mse(X_test,Y_test,W)
        # END TODO

        train_mses.append(train_mse)
        test_mses.append(test_mse)

        # TODO: Update w and b using a single step of ISTA. You are not allowed to use loops here.

        diff = np.matmul(X_train,W) - Y_train
        gradient = (1/X_train.shape[0]) * np.matmul(np.transpose(X_train),diff)
        W_ls = W - lr * gradient
        W_lasso = np.where(abs(W_ls) < _lambda*lr,0,W_ls)
        W_lasso = np.where(W_ls > _lambda*lr,W_ls-_lambda*lr,W_lasso)
        W_lasso = np.where(W_ls < -1*_lambda*lr,W_ls+_lambda*lr,W_lasso)
        # END TODO
        # TODO: Stop the algorithm if the norm between previous W and current W falls below 1e-4
        if float(np.sqrt(np.sum((W-W_lasso)**2,axis=0))) <= 1e-4:
            W = W_lasso
            break
        else:
            W = W_lasso
        # End TODO

    return W, train_mses, test_mses

def experimentbc(X_train, Y_train, X_test, Y_test):
    # lambdas = np.linspace(0.1,6,16)
    lambdas = [0.1, 0.15, 0.2, 0.3, 0.7, 1.5, 1.9, 2.3, 2.7, 3.2, 3.6, 4.0, 4.6, 5.0, 5.5, 6.0]
    lr = 0.001
    max_iter = 10000
    final_trains = []
    final_tests = []
    # start = time.time()
    for mylambda in lambdas:
        W, train_mses_ista, test_mses_ista = ista(X_train, Y_train, X_test, Y_test,_lambda=mylambda,lr=lr,max_iter=max_iter)
        final_trains.append(train_mses_ista[-1])
        final_tests.append(test_mses_ista[-1])
    W, train_mses_ista, test_mses_ista = ista(X_train, Y_train, X_test, Y_test,_lambda=0.2,max_iter=40000)
    W_ridge, train_mses, test_mses = ridge_regression(X_train, Y_train, X_test, Y_test, 10)
    # end = time.time()
    # print(end - start)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(15,5))

    ax1.plot(lambdas,final_trains)
    ax1.plot(lambdas,final_tests)
    ax1.set_title('MSE vs lambda')
    ax1.legend(['Train MSE', 'Test MSE'])
    ax1.set_xlabel('lambda')
    ax1.set_ylabel('MSE')

    ax2.scatter(np.arange(W.shape[0]),W)
    ax2.set_title('ISTA')
    ax2.set_xlabel('index')

    ax3.set_xlabel('index')
    ax3.set_title('Ridge')
    ax3.scatter(np.arange(W_ridge.shape[0]),W_ridge)

    plt.show()

if __name__ == '__main__':
    # Load and split data
    X, Y = load_data2('data2.csv')
    X, Y = preprocess(X, Y)
    X_train, Y_train, X_test, Y_test = split_data(X, Y)

    # TODO: Your code for plots required in Problem 1.2(b) and 1.2(c)
    experimentbc(X_train, Y_train, X_test, Y_test)
    # End TODO
