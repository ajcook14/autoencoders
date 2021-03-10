from net import Net
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

layers = [2, 1, 2]

weights = [np.array([[20, 10]]), np.array([[10], [0]])]
biases = [np.array([-20]), np.array([0, 0])]
parameters = (weights, biases)

net = Net(layers)
test_net = Net(layers, parameters=parameters)



data_size = 100

xx = np.arange(0, 1, 1/data_size)
yy = np.arange(0, 1, 1/data_size)
zz = np.zeros(xx.shape + yy.shape)

training_data = []

# ideally, feedforward should take in a small dataset (2-D np.ndarray), so the following loops can be paralellized
for i in range(data_size):

    for j in range(data_size):

        x = xx[i]
        y = yy[j]

        output = test_net.feedforward(np.array([x, y]))

        training_data.append((np.array([x, y]), output))

        zz[i, j] = output[0]

epochs = 20
minibatch_size = data_size**2
eta = 0.001
#net.SGD(training_data, epochs, minibatch_size, eta)
#print(net.weights, net.biases)

#h = plt.scatter(xx, np.log(100*xx + 1), marker='x')

#plt.show()
fig = plt.figure()
ax = fig.gca(projection='3d')

xx, yy = np.meshgrid(xx, yy, sparse=True)

surf = ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm, linewidth=0, antialiased=False)

ax.set_zlim(0.0, 1.0)
plt.show()
"""
x = np.arange(-1, 1, 200)
y = net.d_sigmoid(x)
h = plt.plot(x, y)
print(net.d_sigmoid(0))
"""
