import numpy as np
import matplotlib.pyplot as plt



def sigmoid(z):

    return(1 / (1 + np.exp(-z) ))

def d_sigmoid(z):

    sig = sigmoid(z)

    return(sig * (1 - sig))

def inv_sigmoid(z):

    return(-np.log( (1/z) - 1))

def d_inv_sigmoid(z):

    return(1 / (z - z**2))



x = np.linspace(0.45, 0.55, 100)
u = 50

b = -u/2
v = 4/( u * d_sigmoid(0.5 * u + b) )
c = -v * sigmoid(0.5 * u + b)

y = v * sigmoid(u * x + b) + c



fig, ax = plt.subplots()

ax.plot(x, y, x, inv_sigmoid(x))

plt.show()
