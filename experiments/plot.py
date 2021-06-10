import numpy as np
import matplotlib.pyplot as plt
from pyibex import *

from net import Net
from interval_bisection import interval_bisection
from diffae import DiffAE
from queue import Queue

def sigmoid(z):

    return(1 / (1 + np.exp(-z) ))

def inv_sigmoid(z):

    return(-np.log( (1/z) - 1))

def d_inv_sigmoid(z):

    return(1 / (z - z**2))

M = 100
N = 100
x = np.linspace(1/M, 1 - 1/M, N)

"""
a1 = sigmoid(7.08104112 * x)
a2 = sigmoid(-8.53176666 * x)

y = (0.37042102 * a1 + -9.46895952 * a2)

inv = inv_sigmoid(x)
"""

L = 2
seed = 0
rng = np.random.default_rng(seed)


#weights = rng.uniform(-5, 5, L)
#biases = rng.uniform(-10, 10, L)

#layers = [1,2,1]

#weights = [np.array([[ 659.53597234], [-127.96921034]]), np.array([[-345.09630499, -526.3422031]])]
#biases = [np.array([-596.31565023,  115.70278649]), np.array([450.93610533])]

#weights = [np.array([[546.0968235], [633.008118 ]]), np.array([[630.35800621, 169.9683145 ]])]
#biases = [np.array([ -53.34973947, -379.57828465]), np.array([-632.18283872])]

weights = [np.array([[ 992.09876479], [-639.18347051]]), np.array([[ 887.88720805, -305.02006012]])]
biases = [np.array([-406.28195719,  357.31351743]), np.array([-583.05239577])]
layers = [1,2,1]

"""
d = 5
layers = [1, d, 1]

weights = [50 * np.ones((d, 1)), np.array([[1.5, 1, 1.5]])]
biases = [50 * np.linspace(-0.2, -0.8, d), np.array([-2])]

weights = [50 * (2**d) * np.ones((d, 1)), d_inv_sigmoid(sigmoid(np.linspace([-d], [d], d).T))]
biases = [50 * (2**d) * -sigmoid(np.linspace(-d, d, d)), -np.array(np.sum(weights[1])/2)]
"""

parameters = (weights, biases)

net = Net(layers, parameters=parameters)

f = DiffAE(net)
queue = Queue()
queue.append(np.array([Interval(0, 1)]))

verified = interval_bisection(f, queue)
print(len(verified))

#print([item[0].mid() for item in verified])

y = np.zeros_like(x)

for i in range(N):

    out = net.feedforward(np.array([x[i]]))

    y[i] = out[0]

"""
i += 1
a = weights[i] * a + biases[i]

a = 0.5 * sigmoid(10*x) - 1
b = sigmoid(2*x)
c = sigmoid(2*(0.5 * sigmoid(10*x) - 1))
"""

fig, ax = plt.subplots(figsize=(8, 8))

#ax.plot(a, x, x, b)
#ax.set_ylim(-2.5, 2.5)
#ax.plot(x, inv_sigmoid(y), x, inv_sigmoid(x))
ax.plot(x, y, x, x)#inv_sigmoid(y), x, inv_sigmoid(x))

plt.show()
