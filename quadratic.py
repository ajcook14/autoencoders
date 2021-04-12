from net import Net
from parameters import Parameters
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import pickle
import gzip
import time

marker_size = mpl.rcParams['lines.markersize'] ** 2

layers = [2, 2, 1, 2, 2]

net = Net(layers)

n = 100

# initialize the data
x = np.arange(0.1, 0.9, 0.8/n)
y = 1 * (x - x**2)

data = np.stack([x, y])

training_data = []

for i in range(n):

    training_data.append( (data[:, i], data[:, i]) )

# train the network
epochs = 5000
size_minibatch = n // 20
size_validation = n // 5
eta = 2
validation_costs = net.SGD(training_data, epochs, size_minibatch, size_validation, eta)

# save the network parameters
fname = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
f = gzip.open(f'./data/quadratic/{fname}', 'wb')

seeds = (net.seed, net.np_seed)
parameters = (net.weights, net.biases)
training_info = ''
params = Parameters(layers, training_data, epochs, size_minibatch, size_validation, eta, seeds=seeds, parameters=parameters, training_info=training_info)

pickle.dump(params, f)
f.close()

# compute the output manifold
output = np.zeros((2, n))

for i in range(n):

    output[:, i] = net.feedforward( data[:, i] )

xo = output[0, :]
yo = output[1, :]

# compute vector field
xi = np.arange(0, 1, 1/n)
yi = np.arange(0, 1, 1/n)

xx, yy = np.meshgrid(xi, yi)
uu, vv = np.meshgrid(xi, yi)

for i in range(n):

    for j in range(n):

        output = net.feedforward( np.array([xx[i, j], yy[i, j]]) )

        uu[i, j] = output[0] - xx[i, j]
        vv[i, j] = output[1] - yy[i, j]

# plot results
validation_epochs = np.arange(epochs, dtype='float64')

fig, ax = plt.subplots(1, 2)
ax[0].plot(validation_costs)

ax[1].scatter(x, y, s=marker_size/4, c='b', label='input')
ax[1].scatter(xo, yo, s=marker_size/4, c='g', label='output')
q = ax[1].quiver(xx, yy, uu, vv)

s = 'architecture = %s, n = %d, epochs = %d, \nsize_minibatch = %d, eta = %f'%\
(str(layers), n, epochs, size_minibatch, eta)
plt.title(s)
ax[1].legend()

plt.show()
