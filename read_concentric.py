
from net import Net
import numpy as np
import gzip
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

marker_size = mpl.rcParams['lines.markersize'] ** 2



f = gzip.open('./data/concentric/20210330_150744', 'rb')
net = pickle.load(f)
f.close()



n = 25
size_validation = (2 * n) // 5

# initialize the data
theta = np.linspace(-np.pi, np.pi, n + size_validation)
x_out = 0.4 * np.cos(theta) + 0.5
y_out = 0.4 * np.sin(theta) + 0.5

x_in = 0.2 * np.cos(theta) + 0.5
y_in = 0.2 * np.sin(theta) + 0.5

data_x = np.hstack([x_out, x_in])
data_y = np.hstack([y_out, y_in])
data = np.stack([data_x, data_y])

# compute output manifold
output = np.zeros((2, 2 * n))

for i in range(2 * n):

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

"""
# compute encoder output
layers = net.layers
encoder = Net( layers[:3], (net.weights[:2], net.biases[:2]) )
zz = np.zeros((m, m))

for i in range(m):

    for j in range(m):

        output = encoder.feedforward( np.array([xx[i, j], yy[i, j]]) )

        zz[i, j] = output[0]
"""

# plot results
fig, ax = plt.subplots()

ax.scatter(data_x, data_y, s=marker_size/4, c='b', label='input')
ax.scatter(xo, yo, s=marker_size/4, c='g', label='output')
q = ax.quiver(xx, yy, uu, vv)

ax.legend()

# a very simple ode solver using Euler's method
def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))

    trace_x = [event.xdata]
    trace_y = [event.ydata]
    eps = 1
    h = 0.01
    while eps > 0.01:
        x_prev = np.array([trace_x[-1], trace_y[-1]])

        output = net.feedforward(x_prev)

        dx = output - x_prev

        x_curr = x_prev + h * dx

        trace_x.append(x_curr[0])
        trace_y.append(x_curr[1])

        eps = np.linalg.norm(dx)

    ax.plot(trace_x, trace_y, c='r')
    plt.show()

cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

"""
# plot the encoder
fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.set_zlim(0.0, 1.0)
plt.show()
"""
