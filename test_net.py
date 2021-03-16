from net import Net
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

layers = [2, 1, 2]

weights = [np.array([[0.9, 0.4]]), np.array([[0.4], [0.0]])]
biases = [np.array([-1.1]), np.array([0.1, 0.2])]
parameters = (weights, biases)
net = Net(layers)

weights = [np.array([[1.0, 0.5]]), np.array([[0.5], [0.0]])]
biases = [np.array([-1.0]), np.array([0.0, 0.0])]
parameters = (weights, biases)
test_net = Net(layers, parameters=parameters)



n = 20

xx = np.arange(0, 1, 1/n)
yy = np.arange(0, 1, 1/n)
zz = np.zeros(xx.shape + yy.shape)

training_data = []

# ideally, feedforward should take in a small dataset (2-D np.ndarray), so the following loops can be paralellized
for i in range(n):

    for j in range(n):

        x = xx[i]
        y = yy[j]

        output = test_net.feedforward(np.array([x, y]))

        training_data.append((np.array([x, y]), output))

        zz[i, j] = output[0]

epochs = 600
data_size = n**2
size_minibatch = data_size // 6
size_validation = data_size // 6
eta = 0.005
validation_costs = net.SGD(training_data, epochs, size_minibatch, size_validation, eta)

validation_epochs = np.array(range(epochs))
plot = plt.plot(validation_epochs, validation_costs)
plt.show()

zz_trained = np.zeros(xx.shape + yy.shape)
for i in range(n):

    for j in range(n):

        x = xx[i]
        y = yy[j]

        output = net.feedforward(np.array([x, y]))

        zz_trained[i, j] = output[0]

print(net.weights, net.biases)

#h = plt.scatter(xx, np.log(100*xx + 1), marker='x')

#plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')

xx, yy = np.meshgrid(xx, yy, sparse=True)

surf = ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm, linewidth=0, antialiased=False)

ax.set_zlim(0.5, 0.6)
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')

xx, yy = np.meshgrid(xx, yy, sparse=True)

surf = ax.plot_surface(xx, yy, zz_trained, cmap=cm.coolwarm, linewidth=0, antialiased=False)

ax.set_zlim(0.5, 0.6)
plt.show()



"""
x = np.arange(-1, 1, 200)
y = net.d_sigmoid(x)
h = plt.plot(x, y)
print(net.d_sigmoid(0))
"""
