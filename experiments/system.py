import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pyibex import *

from interval_bisection import interval_bisection
from net import Net
from diffae import DiffAE
from queue import Queue

def sigmoid(z):

    return(1 / (1 + np.exp(-z) ))

def d_sigmoid(z):

    sig = sigmoid(z)

    return(sig * (1 - sig))

def inv_sigmoid(z):

    return(-np.log( (1/z) - 1))

def d_inv_sigmoid(z):

    return(1 / (z - z**2))

def f(x, v, u, b, c):

    return(v * sigmoid(u * x + b) + c)



x_0 = 1/4
x_2 = 3/4

u = 7
b = 1/10

v = ( (inv_sigmoid(x_2) - inv_sigmoid(x_0)) / sigmoid(u * x_2 + b) ) / (1 - sigmoid(u * x_0 + b) / sigmoid(u * x_2 + b))
c = inv_sigmoid(x_0) - v * sigmoid(u * x_0 + b)

x = np.linspace(0.01, 0.99, 100)
y = f(x, v, u, b, c)



eps = 1e-10 # to stop interval padding from double counting
layers = [1,1,1]
weights = [np.array([[u]]), np.array([[v]])]
biases = [np.array([b + eps]), np.array([c])]
parameters = (weights, biases)

net = Net(layers, parameters=parameters)

g = DiffAE(net)
queue = Queue()
init = np.array([Interval(0, 1)])
queue.append(init)
verified = interval_bisection(g, queue)




fig, ax = plt.subplots()

line, = ax.plot(x, y)
ax.plot(x, inv_sigmoid(x))
ax.set_xlim([0, 1])
ax.set_ylim([-5, 5])

max_val = 0

def animate(b):
    global max_val

    #if b < -u/2 + 2/interval:
    #    ani.event_source.stop()

    v = ( (inv_sigmoid(x_2) - inv_sigmoid(x_0)) / sigmoid(u * x_2 + b) ) / (1 - sigmoid(u * x_0 + b) / sigmoid(u * x_2 + b))
    c = inv_sigmoid(x_0) - v * sigmoid(u * x_0 + b)
    line.set_ydata(f(x, v, u, b, c))

    net.weights = [np.array([[u]]), np.array([[v]])]
    net.biases = [np.array([b + eps]), np.array([c])]

    queue.clean()
    queue.append(init)
    verified = interval_bisection(g, queue)

    if len(verified) > max_val:
        max_val = len(verified)

    plt.title('offset = %1.3f'%(-b/u))

interval = 50

ani = animation.FuncAnimation(fig, animate, np.arange(-u/10, -9*u/10, -1/interval), interval=interval, blit=False)

plt.show()

print(max_val)






