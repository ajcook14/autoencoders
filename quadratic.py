from net import Net
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

marker_size = mpl.rcParams['lines.markersize'] ** 2

layers = [2, 2, 1, 2, 2]

net = Net(layers)

n = 50

x = np.arange(0.1, 0.9, 0.8/n)
y = x - x**2

data = np.stack([x, y])

training_data = []

for i in range(n):

    training_data.append( (data[:, i], data[:, i]) )

epochs = 5000
size_minibatch = n // 5
size_validation = n // 5
eta = 2
validation_costs = net.SGD(training_data, epochs, size_minibatch, size_validation, eta)

validation_epochs = np.arange(epochs, dtype='float64')
plot = plt.plot(validation_costs)
plt.show()

output = np.zeros_like(data)

for i in range(n):

    output[:, i] = net.feedforward( data[:, i] )

xo = output[0, :]
yo = output[1, :]

xx, yy = np.meshgrid(x, y)
uu, vv = np.meshgrid(x, y)

for i in range(n):

    for j in range(n):

        output = net.feedforward( np.array([xx[i, j], yy[i, j]]) )

        uu[i, j] = output[0] - xx[i, j]
        vv[i, j] = output[1] - yy[i, j]

fig, ax = plt.subplots()
ax.scatter(x, y, s=marker_size/4, c='b', label='input')
ax.scatter(xo, yo, s=marker_size/4, c='g', label='output')
q = ax.quiver(x, y, uu, vv)

s = 'architecture = %s, n = %d, epochs = %d, \nsize_minibatch = %d, eta = %f'%\
(str(layers), 2 * n, epochs, size_minibatch, eta)
plt.title(s)
ax.legend()

plt.show()
