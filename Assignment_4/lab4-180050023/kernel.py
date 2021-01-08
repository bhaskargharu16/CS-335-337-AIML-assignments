import numpy as np 

def linear_kernel(X,Y,sigma=None):
	'''Returns the gram matrix for a linear kernel
	
	Arguments:
		X - numpy array of size n x d
		Y - numpy array of size m x d
		sigma - dummy argment, don't use
	Return:
		K - numpy array of size n x m
	''' 
	# TODO 
	# NOTE THAT YOU CANNOT USE FOR LOOPS HERE
	return np.matmul(X,np.transpose(Y))
	# END TODO

def gaussian_kernel(X,Y,sigma=0.1):
	'''Returns the gram matrix for a rbf
	
	Arguments:
		X - numpy array of size n x d
		Y - numpy array of size m x d
		sigma - The sigma value for kernel
	Return:
		K - numpy array of size n x m
	'''
	# TODO
	# NOTE THAT YOU CANNOT USE FOR LOOPS HERE 
	return np.exp(-(np.sum(X**2,axis=1,keepdims=True) + np.sum(Y**2,axis=1) - 2 * np.matmul(X,np.transpose(Y)))/(2*sigma*sigma))
	# END TODO

def my_kernel(X,Y,sigma):
	'''Returns the gram matrix for your designed kernel
	
	Arguments:
		X - numpy array of size n x d
		Y - numpy array of size m x d
		sigma- dummy argment, don't use
	Return:
		K - numpy array of size n x m
	''' 
	# TODO
	# NOTE THAT YOU CANNOT USE FOR LOOPS HERE 
	# return np.exp(-(np.sum(X**2,axis=1,keepdims=True) + np.sum(Y**2,axis=1) - 2 * np.matmul(X,np.transpose(Y)))/(0.05))
	return (1+np.matmul(X,np.transpose(Y)))**4
	# END TODO
