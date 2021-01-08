from numpy import array as nparray
from numpy import savetxt, log
from numpy.random import seed,uniform
import argparse

def modify_uniform(u,lamda=1):
    '''
    This function modifies the given number assumed to be from a uniform distribution 

    TASK - Complete this function
    '''
    u = (-1/lamda) * log(u)
    return u



######################################


def generate_expo(n,lamda=1):
    '''
    Generates n IID samples from an exponential distribution
    '''
    exp = []
    for i in range(n):
        u = uniform(0,1)       # This generates a single number from a [0,1] uniform distribution
        x = modify_uniform(u,lamda=1)
        exp.append(x)
    return nparray(exp)   # This can be used to convert a list to a numpy array.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Exponential Distribution')
    parser.add_argument('--num', type=int, default=100,
                    help='Number of samples to generate')
    parser.add_argument('--seed', type=int, default=42,
                    help='Seed for random generator')
    parser.add_argument('--lamda', type=float, default=1,
                    help='Lambda parameter for the distribution')

    args = parser.parse_args()

    seed(args.seed)

    n = args.num

    exp = generate_expo(n,args.lamda)
    savetxt('problem_1.txt',exp,fmt='%1.4f')
    