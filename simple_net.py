from net import Net
import numpy as np
import matplotlib.pyplot as plt

layers = [1, 4, 1]

net = Net(layers)

weights = [np.array([20]), np.array([20])]
biases = [np.array([0]), np.array([-10])]
parameters = (weights, biases)
test_net = Net([1, 1, 1], parameters=parameters)

n = 50

xx = np.arange(-1, 1, 2/n)
zz = np.arange(xx.shape[0], dtype='float64')

training_data = []

for xx_i in range(xx.shape[0]):

    x = xx[xx_i]

    output = test_net.feedforward(np.array([x]))

    zz[xx_i] = output[0]

    training_data.append( (np.array([x]), output) )


epochs = 5000
size_minibatch = n // 5
size_validation = n // 5
eta = 0.09
validation_costs = net.SGD(training_data, epochs, size_minibatch, size_validation, eta)


rr = np.arange(xx.shape[0], dtype='float64')
for xx_i in range(xx.shape[0]):

    x = xx[xx_i]

    output = net.feedforward(np.array([x]))

    rr[xx_i] = output[0]


validation_epochs = np.arange(epochs, dtype='float64')

fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(validation_epochs, validation_costs)
ax1.set_title('validation loss')

ax2.plot(xx, zz, 'b', xx, rr, 'r')
ax2.set_title('target (blue) vs. trained (red)')

plt.show()

