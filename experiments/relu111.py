"""This experiment attempts to train a basic architecture with relu activations to provide some intuition for fixed points.
Last modified: 22/06/2021
Author: Andrew Cook
"""

import numpy as np
import matplotlib.pyplot as plt

from net import Net
import activations

layers = [1, 1, 1]
seeds = (0, 0)
net = Net(layers, seeds=seeds, activation=activations.relu)

training_data = [(np.array([1.]), np.array([1.])), (np.array([2.]), np.array([2.]))]
epochs = 0
size_minibatch = 2
size_validation = 0
eta = 1

net.SGD(training_data, epochs, size_minibatch, size_validation, eta)

N = 100
x = np.linspace(0, 3, N)
y = np.zeros_like(x)

for i in range(N):

    y[i] = net.feedforward(np.array([x[i]]))

fig, ax = plt.subplots()

ax.plot(x, y)#, x, x)
ax.set_ylim(-0.01, 0.1)

plt.show()
