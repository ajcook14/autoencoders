from net import Net
import activations
import numpy as np
from interval_bisection import *
from queue import Queue
from diffae import DiffAE


import matplotlib.pyplot as plt

#layers = [2, 1, 2]
layers = [1, 2, 1]

seeds = (0, 0)

net = Net(layers, seeds, activation=activations.sigmoid)

data = np.array([0.1, 0.4, 0.6, 0.9]).reshape(1, 4)
n = 4

#x = np.array([0.1, 0.4, 0.6, 0.9])
#y = np.array([0.1, 0.4, 0.6, 0.9])
#data = np.stack([x, y])
#n = data.shape[1]

training_data = [(data[:, i].reshape(layers[0], 1), data[:, i].reshape(layers[-1], 1)) for i in range(n)]

n = len(training_data)

epochs = 80000
size_minibatch = 1
size_validation = n
eta = 1.
separate_validation = False

validation_costs = net.SGD(training_data, epochs, size_minibatch, size_validation, eta, separate_validation)

#init = np.array([Interval(0, 1) for _ in range(layers[0])])
#queue = Queue()
#f = DiffAE(net)
#queue.clean()
#queue.append(init)

#verified = interval_bisection(f, queue)



fig, ax = plt.subplots(1, 2, figsize=(12, 6))

begin = epochs - epochs // 1
m = np.arange(begin, epochs)
ax[0].plot(m, validation_costs[begin:])
x = np.linspace(0, 1, 50)
y = np.zeros_like(x)
for i in range(x.shape[0]):
    y[i] = net.feedforward(np.array( x[i] ).reshape(1, 1))[0, 0] - x[i]
ax[1].plot(x, y, x, np.zeros_like(x))
ax[0].set_title('training validation')
ax[0].set_xlabel('epochs')
ax[0].set_ylabel('loss')

ax[1].set_title(f'$f(x) - x$ for the 1-2-1 architecture trained on\n[0.1, 0.4, 0.6, 0.9] for {epochs} epochs.')
plt.show()


