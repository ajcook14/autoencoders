from net import Net
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import activations

marker_size = mpl.rcParams['lines.markersize'] ** 2



layers = [2,2,2]#[2, 2, 2] commented parameters lead to ocsillatory behaviour with sigmoid activations

sine = True

if sine:
    net = Net(layers, activation=activations.sin)
else:
    net = Net(layers, activation=activations.sigmoid)



# construct the data (a circle)
n = 20
size_validation = 2

theta = np.arange(-np.pi, np.pi, 2*np.pi/(n + size_validation))
if sine:
    data_x = np.cos(theta)
    data_y = np.sin(theta)
else:
    data_x = 0.4 * np.cos(theta) + 0.5
    data_y = 0.4 * np.sin(theta) + 0.5

data = np.stack([data_x, data_y])

training_data = []

for i in range(n):

    training_data.append( (data[:, i], data[:, i]) )

# train the network
epochs = 4000#50
size_minibatch = 5#1
eta = 3.

validation_costs = net.SGD(training_data, epochs, size_minibatch, size_validation, eta)

# compute the output manifold
output = np.zeros((2, n))

for i in range(n):

    output[:, i] = net.feedforward( data[:, i] )

xo = output[0, :]
yo = output[1, :]

# compute vector field
m = 50

if sine:
    xi = np.arange(-2, 2, 4/m)
    yi = np.arange(-2, 2, 4/m)
else:
    xi = np.arange(0, 1, 1/m)
    yi = np.arange(0, 1, 1/m)

xx, yy = np.meshgrid(xi, yi)
uu, vv = np.meshgrid(xi, yi)

for i in range(m):

    for j in range(m):

        output = net.feedforward( np.array([xx[i, j], yy[i, j]]) )

        uu[i, j] = output[0] - xx[i, j]
        vv[i, j] = output[1] - yy[i, j]

# plot results
fig, ax = plt.subplots(1, 2, figsize=(24, 12))
ax[0].plot(validation_costs)

ax[1].scatter(data_x, data_y, s=marker_size/2, c='b', label='input')
ax[1].scatter(xo, yo, s=marker_size/2, c='r', label='output')
q = ax[1].quiver(xx, yy, uu, vv, color='tab:gray')

s = 'architecture = %s, n = %d, epochs = %d, \nsize_minibatch = %d, eta = %f'%\
(str(layers), n, epochs, size_minibatch, eta)
ax[1].set_title(s)
ax[1].legend()

plt.show()
