from net import Net
import activations
import numpy as np
from interval_bisection import *
from queue import Queue
from diffae import DiffAE


import matplotlib.pyplot as plt

layers = [2, 1, 2]
#layers = [1, 2, 1]

seeds = (0, 0)

net = Net(layers, seeds, activation=activations.sigmoid)

#data = np.array([0.1, 0.4, 0.6, 0.9]).reshape(1, 4)
#n = 4

x = np.array([0.1, 0.4, 0.6, 0.9])
y = np.array([0.1, 0.4, 0.6, 0.9])
data = np.stack([x, y])
n = data.shape[1]

training_data = [(data[:, i].reshape(layers[0], 1), data[:, i].reshape(layers[-1], 1)) for i in range(n)]

n = len(training_data)

epochs = 1000
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



# compute vector field
k = 50
xi = np.arange(0, 1, 2/k)
yi = np.arange(0, 1, 2/k)

xx, yy = np.meshgrid(xi, yi)
uu, vv = np.meshgrid(xi, yi)

for i in range(yi.shape[0]):

    for j in range(xi.shape[0]):

        output = net.feedforward( np.array([xx[i, j], yy[i, j]]).reshape(2, 1) )

        #print(output[0].shape)
        #print(float(output[0] - xx[i, j]))
        #quit()
        uu[i, j] = float(output[0] - xx[i, j])
        vv[i, j] = float(output[1] - yy[i, j])

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

#ax[3].scatter(xx, yy, s=marker_size/4, c='b', label='input')
#ax[3].scatter(xo, yo, s=marker_size/4, c='g', label='output')
q = ax[0].quiver(xx, yy, uu, vv, color='tab:gray')

begin = epochs - epochs // 4
m = np.arange(begin, epochs)
ax[1].plot(m, validation_costs[begin:])
x = np.linspace(0, 1, 50)
y = np.zeros_like(x)
#for i in range(x.shape[0]):
#    y[i] = net.feedforward(np.array( x[i] ).reshape(1, 1))[0, 0] - x[i]
#ax[1].plot(x, y, x, np.zeros_like(x))
plt.show()


