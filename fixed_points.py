from net import Net
from interval_bisection import *
from queue import Queue
from diffae import DiffAE

import numpy as np
import time
import pickle
import gzip
import sys
import matplotlib.pyplot as plt



layers = [1, 1, 1]

L = len(layers) - 1

weights = []
biases = []

for i in range(L):

    weights.append( np.zeros((layers[i + 1], layers[i])) )
    biases.append( np.zeros(layers[i + 1]) )

parameters = (weights, biases)

net = Net(layers, parameters=parameters)

weights_lower = -100.0
weights_upper = 100.0
biases_lower = -100.0
biases_upper = 100.0
seed = 0
rng = np.random.default_rng(seed)
init = np.array([Interval(0, 1) for _ in range(layers[0])])
queue = Queue()
fixed_points = []
iteration = 0

try:

    while True:

        print(f'\r{iteration}', end='')
        sys.stdout.flush()

        for i in range(L):

            weights[i] = rng.uniform(weights_lower, weights_upper, (layers[i + 1], layers[i]))
            biases[i] = rng.uniform(biases_lower, biases_upper, layers[i + 1]) # remove this line for zero biases

        f = DiffAE(net)

        queue.clean()
        queue.append(init)

        verified = interval_bisection(f, queue)

        fixed_points.append(len(verified))

        iteration += 1


except KeyboardInterrupt:

    pass

print('')

fname = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
f = gzip.open(f'./data/fixed_points/{fname}', 'wb')

limits = (weights_lower, weights_upper, biases_lower, biases_upper)
# if no limits, assume weights_lower=-20.0, weights_upper=20.0, 
# biases_lower=-10.0, biases_upper=10.0
pickle.dump((fixed_points, layers, seed, limits), f)

f.close()

print(f'saved to file {fname}')

npfixed = np.array(fixed_points)

print(f'min = {npfixed.min()}')
print(f'max = {npfixed.max()}')
print(f'mean = {npfixed.mean()}')
print(f'std = {npfixed.std()}')
print(f'lenfp = {len(fixed_points)}')

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
