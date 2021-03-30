from net import Net
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

marker_size = mpl.rcParams['lines.markersize'] ** 2

layers = [2, 2, 2]

L = len(layers) - 1

weights = []
biases = []

r = 5.0
for i in range(L):

    #weights.append(np.random.rand(layers[i + 1], layers[i]))
    #weights.append(r * np.random.rand(layers[i + 1], layers[i]))
    #weights.append(r * np.ones((layers[i + 1], layers[i])))
    weights.append(r * np.eye(2))

    #biases.append(np.random.rand(layers[i + 1]))
    biases.append(-0.5 * r * np.ones(2))

parameters = (weights, biases)

net = Net(layers, parameters=parameters)

n = 200
m = 50

theta = np.linspace(-np.pi, np.pi, n)
x_out = 0.4 * np.cos(theta) + 0.5
y_out = 0.4 * np.sin(theta) + 0.5

x_in = 0.2 * np.cos(theta) + 0.5
y_in = 0.2 * np.sin(theta) + 0.5

xi = np.hstack([x_out, x_in])
yi = np.hstack([y_out, y_in])

data = np.stack([xi, yi])



output = np.zeros_like(data)

for i in range(2 * n):

    output[:, i] = net.feedforward( data[:, i] )

xo = output[0, :]
yo = output[1, :]

x = np.arange(0, 1, 1/m)
y = np.arange(0, 1, 1/m)

xx, yy = np.meshgrid(x, y)
uu, vv = np.meshgrid(x, y)

for i in range(m):

    for j in range(m):

        output = net.feedforward( np.array([xx[i, j], yy[i, j]]) )

        uu[i, j] = output[0] - xx[i, j]
        vv[i, j] = output[1] - yy[i, j]

fig, ax = plt.subplots()
ax.scatter(xi, yi, s=marker_size/4, c='b', label='input')
ax.scatter(xo, yo, s=marker_size/4, c='g', label='output')
q = ax.quiver(xx, yy, uu, vv)

ax.legend()

plt.show()
