import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import sys

def d(x,y):
    '''
    Given x and y where each is an np arrays of size (dim,1), compute L2 distance between them
    '''
    return float(np.dot(np.transpose(x-y),x-y))


def pairwise_similarity_looped(X,Y):
    '''
    Given X, Y where each is an np array of size (num_points_1,dim) and (num_points_2,dim), 
    return K, an array having size (num_points_1,num_points_2) according to the problem given
    '''
    ## STEP 1 - Initialize K as a numpy array - ADD CODE TO COMPUTE n1, n2 ##
    
    n1 = X.shape[0]
    n2 = Y.shape[0]
    K = np.zeros((n1, n2))

    ## STEP 2 - Loop and compute  -- COMPLETE THE LOOP BELOW ##

    for i in range(n1):
        x = X[i]
        x = x.reshape(x.shape[0],1)
        for j in range (n2):
            y = Y[j]
            y = y.reshape(y.shape[0],1)
            K[i][j] = d(x,y)
    return K 

def pairwise_similarity_vec(X,Y):
    '''
    Given X, Y where each is an np array of size (num_points_1,dim) and (num_points_2,dim), 
    return K, an array having size (num_points_1,num_points_2) according to the problem given

    This problem can be simplified in the following way - 
    Each entry in K has three terms (as seen in problem 2.1 (a)).
    Hence, first  computethe norm for all points, reshape it suitably,
    then compute the dot product.
    All of these can be done by using on the *, np.matmul, np.sum(), and transpose operators.
    '''
    return (X**2).sum(1,keepdims=True) + (Y**2).sum(1) - 2*X.dot(Y.T)

def time_complexity_analysis(func_indicator,variant_indicator):
    if func_indicator == "vectorized" and variant_indicator == "sample_size":
        sample_size = range(100,1000,100)
        time_taken = []
        for size in sample_size:
            X = np.random.normal(0.,1.,size=(size,100))
            Y = np.random.normal(1.,1.,size=(size,100))
            start = time.time()
            K_vec =  pairwise_similarity_vec(X,Y)
            end = time.time()
            time_taken.append(end-start)
        return time_taken
    if func_indicator == "loop" and variant_indicator == "sample_size":
        sample_size = range(100,1000,100)
        time_taken = []
        for size in sample_size:
            X = np.random.normal(0.,1.,size=(size,100))
            Y = np.random.normal(1.,1.,size=(size,100))
            start = time.time()
            K_vec =  pairwise_similarity_looped(X,Y)
            end = time.time()
            time_taken.append(end-start)
        return time_taken
    if func_indicator == "loop" and variant_indicator == "dim":
        dims = range(10,100,10)
        time_taken = []
        for dim in dims:
            X = np.random.normal(0.,1.,size=(100,dim))
            Y = np.random.normal(1.,1.,size=(100,dim))
            start = time.time()
            K_vec =  pairwise_similarity_looped(X,Y)
            end = time.time()
            time_taken.append(end-start)
        return time_taken
    if func_indicator == "vectorized" and variant_indicator == "dim":
        dims = range(10,100,10)
        time_taken = []
        for dim in dims:
            X = np.random.normal(0.,1.,size=(100,dim))
            Y = np.random.normal(1.,1.,size=(100,dim))
            start = time.time()
            K_vec =  pairwise_similarity_vec(X,Y)
            end = time.time()
            time_taken.append(end-start)
        return time_taken

def experiment():
    sample_size = range(100,1000,100)
    dims = range(10,100,10)
    
    a = time_complexity_analysis("vectorized","sample_size")
    fig = plt.figure()
    plt.plot(sample_size,a)
    title = "Vectorized (num points)"
    plt.title(title)
    plt.xlabel('num points')
    plt.ylabel('Time')
    fig.savefig(title+".png")

    print("vectorized sample_size Done")

    a = time_complexity_analysis("vectorized","dim")
    fig = plt.figure()
    plt.plot(dims,a)
    title = "Vectorized (dimensions)"
    plt.title(title)
    plt.xlabel('dim')
    plt.ylabel('Time')
    fig.savefig(title+".png")

    print("vectorized Dim Done")


    a = time_complexity_analysis("loop","dim")
    fig = plt.figure()
    plt.plot(dims,a)
    title = "loop (dimensions)"
    plt.title(title)
    plt.xlabel('dim')
    plt.ylabel('Time')
    fig.savefig(title+".png")

    print("looped dim Done")

    a = time_complexity_analysis("loop","sample_size")
    fig = plt.figure()
    plt.plot(sample_size,a)
    title = "loop (num points)"
    plt.title(title)
    plt.xlabel('num points')
    plt.ylabel('Time')
    fig.savefig(title+".png")

    print("looped sample_size Done")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--num', type=int, default=5,
                    help='Number of samples to generate')
    parser.add_argument('--seed', type=int, default=42,
                    help='Seed for random generator')
    parser.add_argument('--dim', type=int, default=2,
                    help='Lambda parameter for the distribution')


    args = parser.parse_args()


    np.random.seed(42)
    
    X = np.random.normal(0.,1.,size=(args.num,args.dim))
    Y = np.random.normal(1.,1.,size=(args.num,args.dim))

    t1 = time.time()
    K_loop = pairwise_similarity_looped(X,Y)
    t2 = time.time()
    K_vec  = pairwise_similarity_vec(X,Y)
    t3 = time.time()

    assert np.allclose(K_loop,K_vec)   # Checking that the two computations match

    np.savetxt("problem_2_loop.txt",K_loop)
    np.savetxt("problem_2_vec.txt",K_vec)
    print("Vectorized time : {}, Loop time : {}".format(t3-t2,t2-t1))

    experiment()
    