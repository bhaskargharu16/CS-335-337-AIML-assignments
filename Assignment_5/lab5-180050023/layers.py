'''This file contains the implementations of the layers required by your neural network

For each layer you need to implement the forward and backward pass. You can add helper functions if you need, or have extra variables in the init function

Each layer is of the form - 
class Layer():
    def __init__(args):
        *Initializes stuff*

    def forward(self,X):
        # X is of shape n x (size), where (size) depends on layer
        
        # Do some computations
        # Store activations_current
        return X

    def backward(self, lr, activation_prev, delta):
        """
        # lr - learning rate
        # delta - del_error / del_activations_current
        # activation_prev - input activations to this layer, i.e. activations of previous layer
        """
        # Compute gradients wrt trainable parameters
        # Update parameters
        # Compute gradient wrt input to this layer
        # Return del_error/del_activation_prev
'''
import numpy as np
import copy

class FullyConnectedLayer:
    def __init__(self, in_nodes, out_nodes, activation):
        # Method to initialize a Fully Connected Layer
        # Parameters
        # in_nodes - number of input nodes of this layer
        # out_nodes - number of output nodes of this layer
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.activation = activation   # string having values 'relu' or 'softmax', activation function to use
        # Stores the outgoing summation of weights * feautres 
        self.data = None

        # Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
        self.weights = np.random.normal(0,0.1,(in_nodes, out_nodes))    
        self.biases = np.random.normal(0,0.1, (1, out_nodes))
        ###############################################
        # NOTE: You must NOT change the above code but you can add extra variables if necessary 

    def forwardpass(self, X):
        '''
                
        Arguments:
            X  -- activation matrix       :[n X self.in_nodes]
        Return:
            activation matrix      :[n X self.out_nodes]
        '''
        # TODO
        if self.activation == 'relu':
            a = np.matmul(X,self.weights) + self.biases
            self.data = relu_of_X(a)
            return self.data
        elif self.activation == 'softmax':
            a = np.matmul(X,self.weights) + self.biases
            self.data = softmax_of_X(a)
            return self.data

        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()

        return None
        # END TODO      
    def backwardpass(self, lr, activation_prev, delta):
        '''
        # lr - learning rate
        # delta - del_error / del_activations_current  : 
        # activation_prev - input activations to this layer, i.e. activations of previous layer
        '''

        # TODO 
        if self.activation == 'relu':
            # current_layer_activation = np.matmul(activation_prev, self.weights) + self.biases
            gradient = gradient_relu_of_X(self.data, delta)
            res = np.matmul(gradient,self.weights.T)
            self.biases = self.biases - lr * np.sum(gradient, axis=0)
            self.weights = self.weights -  lr * np.matmul(activation_prev.T, gradient)
            return res
        elif self.activation == 'softmax':
            # current_layer_activation = softmax_of_X(np.matmul(activation_prev, self.weights) + self.biases) 
            gradient = gradient_softmax_of_X(self.data, delta)
            res = np.matmul(gradient, self.weights.T)
            self.biases = self.biases -  lr * np.sum(gradient, axis=0)
            self.weights = self.weights -  lr * np.matmul(activation_prev.T, gradient)
            return res
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()
        # END TODO

def helper2(x, field_height, field_width, padding=1, stride=1):
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    # k, i, j = helper1(x.shape, field_height, field_width, padding, stride)
    N = x.shape[0]
    C = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]
    if (H + 2 * padding - field_height) % stride == 0 and (W + 2 * padding - field_height) % stride == 0 :
        p0 = np.tile(np.repeat(np.arange(field_height), field_width), C)
        p1 = stride * np.repeat(np.arange((H + 2 * padding - field_height) // stride + 1), (W + 2 * padding - field_width) // stride + 1)
        l0 = np.tile(np.arange(field_width), field_height * C)
        l1 = stride * np.tile(np.arange((W + 2 * padding - field_width) // stride + 1), (H + 2 * padding - field_height) // stride + 1)
        k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
        cols = x_padded[:, k, p0.reshape(-1, 1) + p1.reshape(1, -1), l0.reshape(-1, 1) + l1.reshape(1, -1)]
        cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
        return cols
    return None

def helper3(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)

    # k, i, j = helper1(x_shape, field_height, field_width, padding, stride)
    if (H + 2 * padding - field_height) % stride == 0 and (W + 2 * padding - field_height) % stride == 0 :
        p0 = np.tile(np.repeat(np.arange(field_height), field_width), C)
        p1 = stride * np.repeat(np.arange((H + 2 * padding - field_height) // stride + 1), (W + 2 * padding - field_width) // stride + 1)
        l0 = np.tile(np.arange(field_width), field_height * C)
        l1 = stride * np.tile(np.arange((W + 2 * padding - field_width) // stride + 1), (H + 2 * padding - field_height) // stride + 1)
        k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
        cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
        cols_reshaped = cols_reshaped.transpose(2, 0, 1)
        np.add.at(x_padded, (slice(None), k, p0.reshape(-1, 1) + p1.reshape(1, -1), l0.reshape(-1, 1) + l1.reshape(1, -1)), cols_reshaped)
        if padding == 0:
            return x_padded
        return x_padded[:, :, padding:-padding, padding:-padding]
    return None
class ConvolutionLayer:
    def __init__(self, in_channels, filter_size, numfilters, stride, activation):
        # Method to initialize a Convolution Layer
        # Parameters
        # in_channels - list of 3 elements denoting size of input for convolution layer
        # filter_size - list of 2 elements denoting size of kernel weights for convolution layer
        # numfilters  - number of feature maps (denoting output depth)
        # stride      - stride to used during convolution forward pass
        # activation  - can be relu or None
        self.in_depth, self.in_row, self.in_col = in_channels
        self.filter_row, self.filter_col = filter_size
        self.stride = stride
        self.activation = activation
        self.out_depth = numfilters
        self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
        self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

        # Stores the outgoing summation of weights * feautres 
        self.data = None
        
        # Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
        self.weights = np.random.normal(0,0.1, (self.out_depth, self.in_depth, self.filter_row, self.filter_col))   
        self.biases = np.random.normal(0,0.1,self.out_depth)
        
    def forwardpass(self, X):
        # INPUT activation matrix       :[n X self.in_depth X self.in_row X self.in_col]
        # OUTPUT activation matrix      :[n X self.out_depth X self.out_row X self.out_col]

        # TODO
        if self.activation == 'relu':
            self.data = (np.matmul(self.weights.reshape(self.out_depth, -1),helper2(X, self.filter_row, self.filter_col, padding=0, stride=self.stride)) + self.biases.reshape(-1,1)).reshape(self.out_depth, self.out_row, self.out_col, -1).transpose(3, 0, 1, 2)
            return relu_of_X(self.data)
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()
        
        ###############################################
        # END TODO
    def backwardpass(self, lr, activation_prev, delta):
        # Input
        # lr : learning rate of the neural network
        # activation_prev : Activations from previous layer
        # delta : del_Error/ del_activation_curr
        # Output
        # new_delta : del_Error/ del_activation_prev
        
        # Update self.weights and self.biases for this layer by backpropagation
        # TODO

        ###############################################
        if self.activation == 'relu':
            # inp_delta = actual_gradient_relu_of_X(self.data, delta)
            # # raise NotImplementedError
            # X_col = 
            del_grad = gradient_relu_of_X(self.data, delta)
            db_grad = np.sum(del_grad, axis=(0, 2, 3))            
            dW_grad = np.matmul(del_grad.transpose(1, 2, 3, 0).reshape(self.out_depth, -1), helper2(activation_prev, self.filter_row, self.filter_col, padding=0, stride=self.stride).T).reshape(self.weights.shape)
            W_reshape = self.weights.reshape(self.out_depth, -1)
            dX_cache = helper3(np.matmul(W_reshape.T, del_grad.transpose(1, 2, 3, 0).reshape(self.out_depth, -1)), activation_prev.shape, self.filter_row, self.filter_col, padding=0, stride=self.stride)
            self.biases = self.biases -  lr * db_grad
            self.weights = self.weights -  lr * dW_grad
            return dX_cache
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()
        ###############################################

        # END TODO
    
class AvgPoolingLayer:
    def __init__(self, in_channels, filter_size, stride):
        # Method to initialize a Convolution Layer
        # Parameters
        # filter_size - list of 2 elements denoting size of kernel weights for convolution layer

        # NOTE: Here we assume filter_size = stride
        # And we will ensure self.filter_size[0] = self.filter_size[1]
        self.in_depth, self.in_row, self.in_col = in_channels
        self.filter_row, self.filter_col = filter_size
        self.stride = stride

        self.out_depth = self.in_depth
        self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
        self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)


    def forwardpass(self, X):
        # print('Forward MP ')
        # Input
        # X : Activations from previous layer/input
        # Output
        # activations : Activations after one forward pass through this layer
        
        # TODO
        return np.mean(helper2(X.reshape(X.shape[0] * self.in_depth, 1, self.in_row, self.in_col), self.filter_row, self.filter_col, padding=0, stride=self.stride), axis=0).reshape(self.out_row, self.out_col, X.shape[0], self.out_depth).transpose(2, 3, 0, 1)
        # END TODO
        ###############################################
        
    def backwardpass(self, alpha, activation_prev, delta):
        # Input
        # lr : learning rate of the neural network
        # activation_prev : Activations from previous layer
        # activations_curr : Activations of current layer
        # delta : del_Error/ del_activation_curr
        # Output
        # new_delta : del_Error/ del_activation_prev
        
        # TODO
        X = helper2(activation_prev.reshape(activation_prev.shape[0] * self.in_depth, 1, self.in_row, self.in_col), self.filter_row, self.filter_col, padding=0, stride=self.stride)
        dX_grad = np.zeros(X.shape,dtype = X.dtype)
        dX_grad[:, :] = delta.transpose(2, 3, 0, 1).ravel() / (self.filter_row * self.filter_col)
        return helper3(dX_grad, (activation_prev.shape[0] * self.in_depth, 1, self.in_row, self.in_col), self.filter_row, self.filter_col, padding=0, stride=self.stride).reshape(activation_prev.shape)
        # END TODO
        ###############################################



class MaxPoolingLayer:
    def __init__(self, in_channels, filter_size, stride):
        # Method to initialize a Convolution Layer
        # Parameters
        # filter_size - list of 2 elements denoting size of kernel weights for convolution layer

        # NOTE: Here we assume filter_size = stride
        # And we will ensure self.filter_size[0] = self.filter_size[1]
        self.in_depth, self.in_row, self.in_col = in_channels
        self.filter_row, self.filter_col = filter_size
        self.stride = stride

        self.out_depth = self.in_depth
        self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
        self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)


    def forwardpass(self, X):
        # print('Forward MP ')
        # Input
        # X : Activations from previous layer/input
        # Output
        # activations : Activations after one forward pass through this layer
        
        # TODO
        X_mod = helper2(X.reshape(X.shape[0] * self.in_depth, 1, self.in_row, self.in_col), self.filter_row, self.filter_col, padding=0, stride=self.stride)
        return (X_mod[np.argmax(X_mod, axis=0), range(np.argmax(X_mod, axis=0).size)]).reshape(self.out_row, self.out_col, X.shape[0], self.out_depth).transpose(2, 3, 0, 1)
        # END TODO
        ###############################################
        
    def backwardpass(self, alpha, activation_prev, delta):
        # Input
        # lr : learning rate of the neural network
        # activation_prev : Activations from previous layer
        # activations_curr : Activations of current layer
        # delta : del_Error/ del_activation_curr
        # Output
        # new_delta : del_Error/ del_activation_prev
        
        # TODO
        X = helper2(activation_prev.reshape(activation_prev.shape[0] * self.in_depth, 1, self.in_row, self.in_col), self.filter_row, self.filter_col, padding=0, stride=self.stride)
        dX_grad = np.zeros(X.shape,dtype = X.dtype) 
        dX_grad[np.argmax(X, axis=0), range(np.argmax(X, axis=0).size)] = delta.transpose(2, 3, 0, 1).ravel()
        return helper3(dX_grad, (activation_prev.shape[0] * self.in_depth, 1, self.in_row, self.in_col), self.filter_row, self.filter_col, padding=0, stride=self.stride).reshape(activation_prev.shape)
        # END TODO
        ###############################################


# Helper layer to insert between convolution and fully connected layers
class FlattenLayer:
    def __init__(self):
        pass
    
    def forwardpass(self, X):
        # TODO
        # print(X.shape)
        n = X.shape[0]
        return X.reshape(n,-1)
    def backwardpass(self, lr, activation_prev, delta):
        return delta.reshape(activation_prev.shape)
        # END TODO

# Function for the activation and its derivative
def relu_of_X(X):

    # Input
    # data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
    # Returns: Activations after one forward pass through this relu layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation relu
    # TODO
    mycopy = copy.deepcopy(X)
    mycopy = np.where(mycopy <= 0.0, 0.0, mycopy)
    return mycopy
    # END TODO 
    
def gradient_relu_of_X(X, delta):
    # Input
    # Note that these shapes are specified for FullyConnectedLayers, the function also needs to work with ConvolutionalLayer
    # data : Output from next layer/input | shape: batchSize x self.out_nodes
    # delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
    # Returns: Current del_Error to pass to current layer in backward pass through relu layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation relu amd during backwardpass
    
    # TODO
    mycopy = np.zeros_like(X)
    # mycopy = np.where(X > 0.0, delta,mycopy)
    # mycopy = np.where(mycopy <= 0.0,0.0,1.0) * delta
    mycopy[X > 0.0] = 1.0
    return mycopy*delta
    # END TODO

def softmax_of_X(X):
    # Input
    # data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
    # Returns: Activations after one forward pass through this softmax layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation softmax
    
    # TODO
    exp = np.exp(X)
    return exp/np.sum(exp,axis = 1,keepdims=True)
    # END TODO  
def gradient_softmax_of_X(X, delta):
    # Input
    # data : Output from next layer/input | shape: batchSize x self.out_nodes
    # delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
    # Returns: Current del_Error to pass to current layer in backward pass through softmax layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation softmax amd during backwardpass
    # Hint: You might need to compute Jacobian first
    # TODO
    Y = np.zeros(X.shape)
    for i in range(X.shape[0]):
        vec = X[i,:].reshape(-1,1)
        Jacobian = (np.eye(X.shape[1]) * vec) -  np.matmul(vec, vec.T) 
        Y[i:i+1,:] = np.matmul(Jacobian, delta[i,:].reshape(-1,1)).reshape(-1)
    return Y
    # END TODO
