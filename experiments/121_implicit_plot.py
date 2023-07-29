from matplotlib import pyplot as plt
import numpy as np

delta = 0.05
r = 20.
x = np.arange(0., 1., 0.05)
b = np.arange(-r, 0., delta)
d = np.arange(-r, 0., delta)

X, B, D = np.meshgrid(x, b, d)

sigmoid = np.vectorize(lambda x: 1/(1 + np.exp(-x)))
dsigmoid = lambda x: sigmoid(x)*(1 - sigmoid(x))
h = sigmoid(D * sigmoid(B * X)) #1/(1 + np.exp(-D * (1/(1 + np.exp(-B))))) - X
dh = B * D * dsigmoid(B * X) * dsigmoid(D * sigmoid(B * X))

m = np.zeros((h.shape[0], h.shape[2]))

tol = 1e-3

for i in range(m.shape[0]):

    for j in range(m.shape[1]):

        if np.min(np.abs(h)[i, :, j]) < tol and np.min(np.abs(dh)[i, :, j]) < tol:

            m[i, j] = 1

        else:

            m[i, j] = 0

plt.matshow(m)
plt.show()
