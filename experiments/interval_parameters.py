import numpy as np
from pyibex import *
from net import Net
import activations
from diffae import DiffAE
from interval_bisection import *
from queue import Queue


layers = [2, 1, 2]

L = len(layers) - 1

weights = []
biases = []

weights.append(np.array([[Interval(0.9, 1.1), Interval(0.9, 1.1)]]))
weights.append(np.array([[Interval(0.9, 1.1)], [Interval(0.9, 1.1)]]))

biases.append(np.array([[Interval(-0.1, 0.1)]]))
biases.append(np.array([[Interval(-0.1, 0.1)], [Interval(-0.1, 0.1)]]))

parameters = (weights, biases)

net = Net(layers, parameters=parameters, activation=activations.sigmoid)

init = np.array([Interval(0, 1), Interval(0, 1)])

f = DiffAE(net)

queue = Queue()
queue.append(init)

print(interval_bisection(f, queue))
