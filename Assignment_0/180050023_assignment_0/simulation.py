import numpy as np
import matplotlib.pyplot as plt

def toss_coin(ph):
    return np.random.uniform() < ph

def experiment(ph):
    heads = 0
    counter = 0
    while True:
        if heads == 2:
            return counter
        elif toss_coin(ph) :
            heads += 1
            counter += 1
        else:
            heads = 0
            counter += 1

def simulation_helper(num_experiments):
    observed_values = []
    for i in range(num_experiments):
        observed_values.append(experiment(0.75))
    avg_observed_value = np.mean(observed_values)
    return avg_observed_value

def simulation():
    observations = {10:[],100:[],1000:[],10000:[]}
    n = [10,100,1000,10000]
    for i in range(10):
        for num_experiments in n:
            observed = simulation_helper(num_experiments)
            observations[num_experiments].append(observed)
    avg_expectations = [np.mean(observations[10]),np.mean(observations[100]),np.mean(observations[1000]),np.mean(observations[10000])]
    errors = [x - 3.111111 for x in avg_expectations]
    return avg_expectations, errors

if __name__ == '__main__':
    np.random.seed(42)
    n = [10,100,1000,10000]
    avg_expectations, errors = simulation()
    fig = plt.figure()
    title = "observed_expectations_and_errorbars"
    plt.title(title)
    plt.xlabel("log(n) (n steps)")
    plt.ylabel("observed toss value (avg of 10)")
    plt.xscale("log")
    plt.errorbar(n,avg_expectations,yerr=errors,fmt='o',color='black',ecolor='gray',elinewidth=3,capsize=0)
    fig.savefig(title+".png")
    print("Simulation complete please check the saved plot image :  {}".format(title+".png"))




        
    
