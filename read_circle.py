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
f = gzip.open(f'./data/circle/{file_name}', 'rb')
#f = gzip.open('./data/circle/20210413_222431', 'rb') #20210414_115426', 'rb')
params = pickle.load(f)
f.close()

if int(file_name[:4]) <= 2021:
    for i in range(len(params.biases)):
        params.biases[i] = params.biases[i][np.newaxis].T

parameters = (params.weights, params.biases)
net = Net(params.layers, parameters=parameters)
print(f'seeds = {(params.seed, params.np_seed)}')
print(net.layers)

# initialize the data
assert(isinstance(params.training_data, list))

n = len(params.training_data)

data = np.zeros((2, n))

for i in range(len(params.training_data)):

    point = params.training_data[i]

    data[:, i] = point[0]

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

# interval bisection and newton
f = DiffAE(net)

u = Interval(0, 1)
v = Interval(0, 1)
init = np.array([u, v])
queue = Queue()
queue.append(init)

verified = interval_bisection(f, queue)

print('number of verified intervals = %d'%len(verified))

fig, ax = plt.subplots(figsize=(12, 12))

#ax.set_xlim([0, 1])
#ax.set_ylim([0, 1])

rectangles(ax, verified)

# plot results
x = data[0,:]
y = data[1,:]
ax.scatter(x, y, s=marker_size*2, c='b', marker="+", label='input')
ax.scatter(xo, yo, s=marker_size*2, c='g', marker="x", label='output')
q = ax.quiver(xx, yy, uu, vv, color='tab:gray')

ax.legend()

ax.set_xlabel("$x$", fontsize=18)
ax.set_ylabel("$y$", fontsize=18)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='both', which='major', labelsize=15)
ax.tick_params(axis='both', which='minor', labelsize=15)

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
#plt.show()
#plt.savefig('./figures/circle.png')
#plt.savefig('./figures/circle/tol=%f.png'%tol)
plt.savefig(f'./figures/circle/{file_name}_newton.png')

"""
# plot the encoder
fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.set_zlim(0.0, 1.0)
plt.show()
"""
