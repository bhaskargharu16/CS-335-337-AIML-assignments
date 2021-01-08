import numpy as np
import argparse

def get_data(dataset):
	datasets = ['D1', 'D2']
	assert dataset in datasets, "Dataset {dataset} not supported. Supported datasets {datasets}"
	X_train = np.loadtxt(f'data/{dataset}/training_data')
	Y_train = np.loadtxt(f'data/{dataset}/training_labels', dtype=int)
	X_test = np.loadtxt(f'data/{dataset}/test_data')
	Y_test = np.loadtxt(f'data/{dataset}/test_labels', dtype=int)

	return X_train, Y_train, X_test, Y_test

def get_features(x):
	'''
	Input:
	x - numpy array of shape (2500, )

	Output:
	features - numpy array of shape (D, ) with D <= 5
	'''
	### TODO
	image_matrix = x.reshape(50,50)
	features = np.zeros(4).astype(float)
	features[2] = 1e9
	features[3] = 1
	for i in range(50):
		for j in range(50):
			if image_matrix[i][j]: #pixel is black
				distance_from_center = ((i-25)**2 + (j-25)**2)**0.5
				is_border = not(image_matrix[i-1][j]) or not(image_matrix[i+1][j]) or not(image_matrix[i][j-1]) or not(image_matrix[i][j+1])
				if is_border:
					features[0] += 1 # increase perimneter
					features[1] = max(features[1],distance_from_center)#update max distance of border point
					features[2] = min(features[2],distance_from_center)#update min distance of border point
	return features
	### END TODO

class Perceptron():
	def __init__(self, C, D):
		'''
		C - number of classes
		D - number of features
		'''
		self.C = C
		self.weights = np.zeros((C, D))
		
	def pred(self, x):
		'''
		x - numpy array of shape (D,)
		'''
		### TODO: Return predicted class for x
		return np.argmax(np.matmul(self.weights,x.reshape(x.shape[0],1)),axis=0)
		### END TODO

	def train(self, X, Y, max_iter=6):
		for iter in range(max_iter):
			for i in range(X.shape[0]):
				### TODO: Update weights
				c = self.pred(X[i])
				if c == Y[i]:
					continue
				self.weights[Y[i]] += X[i]
				self.weights[c] -= X[i]
				### END TODO
			# print(f'Train Accuracy at iter {iter} = {self.eval(X, Y)}')

	def eval(self, X, Y):
		n_samples = X.shape[0]
		correct = 0
		for i in range(X.shape[0]):
			if self.pred(X[i]) == Y[i]:
				correct += 1
		return correct/n_samples

if __name__ == '__main__':
	X_train, Y_train, X_test, Y_test = get_data('D2')

	X_train = np.array([get_features(x) for x in X_train])
	X_test = np.array([get_features(x) for x in X_test])

	C = max(np.max(Y_train), np.max(Y_test))+1
	D = X_train.shape[1]

	perceptron = Perceptron(C, D)

	perceptron.train(X_train, Y_train)
	acc = perceptron.eval(X_test, Y_test)
	print(f'Test Accuracy: {acc}')
