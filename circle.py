from net import Net
from parameters import Parameters
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import pickle
import gzip
import time
import argparse

import activations

marker_size = mpl.rcParams['lines.markersize'] ** 2


parser = argparse.ArgumentParser()
parser.add_argument('--seed', metavar='seed', required=False, type=int, nargs='*',
                    help='pseudorandom seed for reproducibility')
parser.add_argument('--n', metavar='n', required=False, type=int, nargs='*',
                    help='size of training dataset')
parser.add_argument('--k', metavar='k', required=False, type=int, nargs='*',
                    help='number of segments')

args = parser.parse_args()

# initialize the autoencoder
layers = [2, 2, 1, 2, 2]

seeds = (args.seed[0], args.seed[0] + 1)

net = Net(layers, seeds, activation=activations.sigmoid)

# initialize the data
n = args.n[0]   # number of data points
k = args.k[0]   # number of segments
assert(n % k == 0)
l = n // k      # points per segment

points = np.arange(-np.pi, np.pi, 2 * np.pi / (2 * n))
theta = np.zeros(n)
for i in range(k):
    for j in range(l):
        theta[l * i + j] = points[2 * l * i + j]

data_x = 0.4 * np.cos(theta) + 0.5
data_y = 0.4 * np.sin(theta) + 0.5

data = np.stack([data_x, data_y])

training_data = []

for i in range(n):

    training_data.append( (data[:, i], data[:, i]) )

# train the network
epochs = 5000
size_validation = n#// 5
size_minibatch = 1#(2 * n) // 25
eta = 2
separate_validation = False
validation_costs = net.SGD(training_data, epochs, size_minibatch, size_validation, eta, separate_validation=separate_validation)

# save the network parameters
fname = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
f = gzip.open(f'./data/circle/{fname}', 'wb')

seeds = (net.seed, net.np_seed)
parameters = (net.weights, net.biases)
training_info = f'separate_validation={separate_validation}'
params = Parameters(layers, training_data, epochs, size_minibatch, size_validation, eta, seeds=seeds, parameters=parameters, training_info=training_info)

pickle.dump(params, f)
f.close()

print(f'saved to file {fname}')

# compute output manifold
output = np.zeros((2, n))

for i in range(n):

    output[:, i] = net.feedforward( data[:, i] )

xo = output[0, :]
yo = output[1, :]

# compute vector field
m = 50

xi = np.arange(0, 1, 1/m)
yi = np.arange(0, 1, 1/m)

xx, yy = np.meshgrid(xi, yi)
uu, vv = np.meshgrid(xi, yi)

for i in range(m):

    for j in range(m):

        output = net.feedforward( np.array([xx[i, j], yy[i, j]]) )

        uu[i, j] = output[0] - xx[i, j]
        vv[i, j] = output[1] - yy[i, j]

# compute encoder output
"""
encoder = Net( layers[:3], parameters=(net.weights[:2], net.biases[:2]) )
zz = np.zeros((m, m))

for i in range(m):

    for j in range(m):

        output = encoder.feedforward( np.array([xx[i, j], yy[i, j]]) )

        zz[i, j] = output[0]
"""

# plot results
validation_epochs = np.arange(epochs, dtype='float64')

fig, ax = plt.subplots(1, 2, figsize=(24, 12))
ax[0].plot(validation_costs)

ax[1].scatter(data_x, data_y, s=marker_size/4, c='b', label='input')
ax[1].scatter(xo, yo, s=marker_size/4, c='g', label='output')
q = ax[1].quiver(xx, yy, uu, vv)

s = 'architecture = %s, n = %d, epochs = %d, \nsize_minibatch = %d, eta = %f'%\
(str(layers), n, epochs, size_minibatch, eta)
ax[1].set_title(s)
ax[1].legend()

plt.savefig(f'./figures/circle/{fname}.png')

# plot the encoder
"""
fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.set_zlim(0.0, 1.0)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
plt.show()
"""
