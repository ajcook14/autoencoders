from net import Net
from interval_bisection import *
from queue import Queue
from diffae import DiffAE
import activations

import numpy as np
import time
import pickle
import gzip
import sys
import matplotlib.pyplot as plt


def check_fixed_points(layers,
    weights_lower=-100.0,
    weights_upper=100.0,
    biases_lower=-100.0,
    biases_upper=100.0,
    seed=0,
    samples=300000):

    L = len(layers) - 1

    weights = []
    biases = []

    for i in range(L):

        weights.append( np.zeros((layers[i + 1], layers[i])) )
        biases.append( np.zeros(layers[i + 1]) )

    parameters = (weights, biases)

    activation = activations.relu
    net = Net(layers, parameters=parameters, activation=activation)

    rng = np.random.default_rng(seed)
    init = np.array([Interval(0, 1) for _ in range(layers[0])])
    queue = Queue()
    fixed_points = []

    try:

        for iteration in range(samples):

            print(f'\r{iteration} of {samples}', end='')
            sys.stdout.flush()

            for i in range(L):

                weights[i] = rng.uniform(weights_lower, weights_upper, (layers[i + 1], layers[i]))
                biases[i] = rng.uniform(biases_lower, biases_upper, layers[i + 1]) # remove this line for zero biases

            f = DiffAE(net)

            queue.clean()
            queue.append(init)

            verified = interval_bisection(f, queue)

            if verified == -1:

                print('')
                print(f'iteration {iteration} unverified')

                fixed_points.append(-1)

            else:

                fixed_points.append(len(verified))

    except KeyboardInterrupt:

        print('')

        return(-1)


    fname = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    fname += '_' + '-'.join(map(lambda x: str(x), layers))
    
    if activation == activations.sigmoid:

        aname = 'sigmoid'

    elif activation == activations.relu:


        aname = 'relu'

    f = gzip.open(f'./data/fixed_points/layers/{aname}/{fname}', 'wb')

    print(f'len(fixed_points) = {len(fixed_points)}')

    limits = (weights_lower, weights_upper, biases_lower, biases_upper)
    # if no limits, assume weights_lower=-20.0, weights_upper=20.0, 
    # biases_lower=-10.0, biases_upper=10.0
    pickle.dump((fixed_points, layers, seed, limits, activation), f)

    f.close()

    print(f'saved to file {fname}')

    return(0)



def main():

    hidden = 2

    try:

        while hidden == 2:

            layers = [1, hidden, 1]

            if check_fixed_points(layers,
                weights_lower=-10.0,
                weights_upper=10.0,
                biases_lower=-1.0,
                biases_upper=1.0,
                seed=0,#0,
                samples=300000) < 0:

                break

            hidden += 1

        return(0)

    except KeyboardInterrupt:

        print('Interrupted outside main loop!')

        return(-1)

"""
print(net.weights)
x = np.arange(0.0, 1.0, 1.0/100)
y = np.zeros_like(x)

for i in range(100):

    y[i] = net.feedforward(np.array([x[i]])) - np.array([x[i]])

fig, ax = plt.subplots()

ax.plot(x, y)

plt.show()
"""

if __name__ == '__main__':

    main()

