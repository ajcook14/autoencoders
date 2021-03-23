from net import Net
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

marker_size = mpl.rcParams['lines.markersize'] ** 2

layers = [2, 1, 3, 2]

net = Net(layers)

n = 25
size_validation = (2 * n) // 5

theta = np.linspace(-np.pi, np.pi, n + size_validation)
x_out = 0.4 * np.cos(theta) + 0.5
y_out = 0.4 * np.sin(theta) + 0.5

x_in = 0.2 * np.cos(theta) + 0.5
y_in = 0.2 * np.sin(theta) + 0.5

data_x = np.hstack([x_out, x_in])
data_y = np.hstack([y_out, y_in])
data = np.stack([data_x, data_y])

training_data = []

for i in range(2 * n):

    training_data.append( (data[:, i], data[:, i]) )

epochs = 10000
size_minibatch = 1#(2 * n) // 25
eta = 2
validation_costs = net.SGD(training_data, epochs, size_minibatch, size_validation, eta)

validation_epochs = np.arange(epochs, dtype='float64')
plot = plt.plot(validation_costs)
plt.show()

output = np.zeros_like(data)

for i in range(2 * n):

    output[:, i] = net.feedforward( data[:, i] )

out_x = output[0, :]
out_y = output[1, :]

fig, ax = plt.subplots()
ax.scatter(data_x, data_y, s=marker_size/4, c='b', label='input')
ax.scatter(out_x, out_y, s=marker_size/4, c='g', label='output')

s = 'architecture = %s, n = %d, epochs = %d, \nsize_minibatch = %d, eta = %f'%\
(str(layers), 2 * n, epochs, size_minibatch, eta)
plt.title(s)
ax.legend()

plt.show()
