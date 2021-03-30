from net import Net
import numpy as np

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10



layers = [3, 3, 2, 3, 3]

net = Net(layers)

n = 50

theta = np.linspace(-np.pi, np.pi, n)
z = np.linspace(0.1, 0.9, n)
x = 0.4 * np.sin(theta) + 0.5
y = 0.4 * np.cos(theta) + 0.5

data = np.stack([x, y, z])
training_data = []

for i in range(n):

    training_data.append( (data[:,i], data[:,i]) )

'''
epochs = 1000
size_minibatch = 10#n // 5
size_validation = n // 5
eta = 0.2
validation_costs = net.SGD(training_data, epochs, size_minibatch, size_validation, eta)

validation_epochs = np.arange(epochs, dtype='float64')
plot = plt.plot(validation_costs)
plt.show()
'''

output = np.zeros_like(data)

for i in range(n):

    output[:,i] = net.feedforward( data[:,i] )

xo = output[0, :]
yo = output[1, :]
zo = output[2, :]

fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.plot(x, y, z, 'b', label='helix')
ax.plot(xo, yo, zo, 'g', label='output helix')
ax.legend()

'''
s = 'architecture = %s, n = %d, epochs = %d, \nsize_minibatch = %d, eta = %f'%\
(str(layers), n, epochs, size_minibatch, eta)
plt.title(s)
'''

plt.show()
