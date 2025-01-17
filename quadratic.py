from net import Net
from parameters import Parameters
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import pickle
import gzip
import time
import argparse

import activations

marker_size = mpl.rcParams['lines.markersize'] ** 2


parser = argparse.ArgumentParser()
parser.add_argument('--f', metavar='YYYYMMDD_HHMMSS', required=False, type=str, nargs='+',
                    help='file name of the pickled parameters object')
parser.add_argument('--s', metavar='save', required=False, type=bool, nargs='*',
                    help='save the trained network in a parameters object')
parser.add_argument('--seed', metavar='seed', required=False, type=int, nargs='*',
                    help='pseudorandom seed for reproducibility')
parser.add_argument('--n', metavar='n', required=False, type=int, nargs='*',
                    help='size of training dataset')

args = parser.parse_args()

# initialize the autoencoder
if args.f is None:

    layers = [2, 2, 1, 2, 2]

    seeds = (args.seed[0], args.seed[0] + 1)

    net = Net(layers, seeds, activation=activations.sigmoid)

else:

    file_name = args.f[0]

    f = gzip.open(f'./data/quadratic/{file_name}', 'rb')
    params = pickle.load(f)
    f.close()

    seeds = (params.seed, params.np_seed)

    layers = params.layers
    net = Net(layers, seeds=seeds)

# initialize the data

if args.f is None or not isinstance(params.training_data, list):

    k = 3 # number of segments
    m = args.n[0] // k # points per segment
    n = k * m

    a = 0.1
    b = 0.9
    space = (b - a)/(2 * k - 1)

    x = np.hstack([np.arange(a + (2 * i) * space, a + (2 * i + 1) * space, (b - a)/(2 * k - 1)/m) for i in range(k)])
    y = 1 * (x - x**2)

    data = np.stack([x, y])

    training_data = []

    for i in range(n):

        training_data.append( (data[:, i], data[:, i]) )

else:

    n = len(params.training_data)

    data = np.zeros((2, n))

    for i in range(len(params.training_data)):

        point = params.training_data[i]

        data[:, i] = point[0]

    training_data = params.training_data

# train the network
if args.f is None:

    epochs = 5000
    size_minibatch = 1 #n // 20
    size_validation = n#// 5
    separate_validation = False
    eta = 4.

else:

    epochs = params.epochs
    size_minibatch = params.size_minibatch
    size_validation = params.size_validation
    eta = params.eta

validation_costs = net.SGD(training_data, epochs, size_minibatch, size_validation, eta, separate_validation=separate_validation)

# save the network parameters
if not args.s is None:

    if args.f is None:

        fname = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
        f = gzip.open(f'./data/quadratic/{fname}', 'wb')

    else:

        fname = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
        fname = f'{file_name}_{fname}'
        f = gzip.open(f'./data/quadratic/{fname}', 'wb')

    seeds = (net.seed, net.np_seed)
    parameters = (net.weights, net.biases)
    training_info = f'separate_validation={separate_validation}'
    params = Parameters(layers, training_data, epochs, size_minibatch, size_validation, eta, seeds=seeds, parameters=parameters, training_info=training_info)

    pickle.dump(params, f)
    f.close()

    print(f'saved to file {fname}')

# compute the output manifold
output = np.zeros((2, n))

for i in range(n):

    output[:, i] = net.feedforward( data[:, i] )

xo = output[0, :]
yo = output[1, :]

# compute vector field
grid_size = 50
xi = np.arange(0, 1, 1/grid_size)
yi = np.arange(0, 1, 1/grid_size)

xx, yy = np.meshgrid(xi, yi)
uu, vv = np.meshgrid(xi, yi)

for i in range(grid_size):

    for j in range(grid_size):

        output = net.feedforward( np.array([xx[i, j], yy[i, j]]) )

        uu[i, j] = output[0] - xx[i, j]
        vv[i, j] = output[1] - yy[i, j]

# plot results
validation_epochs = np.arange(epochs, dtype='float64')

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(validation_costs)
ax[0].set_title('training validation')
ax[0].set_xlabel('epochs')
ax[0].set_ylabel('loss')

x = data[0,:]
y = data[1,:]
ax[1].scatter(x, y, s=marker_size/4, c='b', label='input')
ax[1].scatter(xo, yo, s=marker_size/4, c='g', label='output')
q = ax[1].quiver(xx, yy, uu, vv)

s = 'architecture = %s, n = %d, epochs = %d, \nsize_minibatch = %d, eta = %f'%\
(str(layers), n, epochs, size_minibatch, eta)
plt.title(s)
ax[1].legend()

if not args.s is None:

    plt.savefig(f'./figures/quadratic/{fname}')
