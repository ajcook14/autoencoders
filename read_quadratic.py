from net import Net
from parameters import Parameters
import numpy as np
import gzip
import pickle
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from interval_bisection import *
from queue import Queue
from diffae import DiffAE

marker_size = mpl.rcParams['lines.markersize'] ** 2

parser = argparse.ArgumentParser()
parser.add_argument('file_name', metavar='YYYYMMDD_HHMMSS', type=str, nargs='+',
                    help='file name of the pickled parameters object')

args = parser.parse_args()

file_name = args.file_name[0]
f = gzip.open(f'./data/quadratic/{file_name}', 'rb')
params = pickle.load(f)
f.close()

parameters = (params.weights, params.biases)
net = Net(params.layers, parameters=parameters)
print(params.layers)

# initialize the data
if isinstance(params.training_data, list):

    n = len(params.training_data)

    data = np.zeros((2, n))

    for i in range(len(params.training_data)):

        point = params.training_data[i]

        data[:, i] = point[0]

else:

    n = 100

    x = np.arange(0.1, 0.9, 0.8/n)
    y = 1 * (x - x**2)

    data = np.stack([x, y])

# compute output manifold
output = np.zeros((2, n))

for i in range(n):

    output[:, i] = net.feedforward( data[:, i] )

xo = output[0, :]
yo = output[1, :]

# compute vector field
xi = np.arange(0, 1, 2/n)
yi = np.arange(0, 0.5, 2/n)

m = yi.shape[0]

xx, yy = np.meshgrid(xi, yi)
uu, vv = np.meshgrid(xi, yi)

for i in range(m):

    for j in range(xi.shape[0]):

        output = net.feedforward( np.array([xx[i, j], yy[i, j]]) )

        uu[i, j] = output[0] - xx[i, j]
        vv[i, j] = output[1] - yy[i, j]

# interval bisection and newton
f = DiffAE(net)

u = Interval(0, 1)
v = Interval(0, 1)
init = np.array([u, v])
queue = Queue()
queue.append(init)

verified = interval_bisection(f, queue)

fig, ax = plt.subplots(figsize=(12, 6))

#ax.set_xlim([0, 1])
#ax.set_ylim([0, 1])

rectangles(ax, verified)

# plot results
x = data[0,:]
y = data[1,:]
ax.scatter(x, y, s=marker_size/4, c='b', label='input')
ax.scatter(xo, yo, s=marker_size/4, c='g', label='output')
q = ax.quiver(xx, yy, uu, vv, color='tab:gray')

ax.legend()
plt.title('number of verified intervals = %d'%len(verified))
ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

# a very simple ode solver using Euler's method
def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))

    trace_x = [event.xdata]
    trace_y = [event.ydata]
    eps = 1
    h = 0.01
    while eps > 0.001:
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
#plt.show()
plt.savefig(f'./figures/quadratic/{file_name}_newton.png')
