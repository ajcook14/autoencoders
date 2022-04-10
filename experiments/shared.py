import numpy as np
from interval_bisection import interval_bisection 
from queue import Queue
from pyibex import *
import matplotlib.pyplot as plt

vtanh = np.vectorize(tanh)
vcosh = np.vectorize(cosh)

class Tanh():

    def __init__(self, a, b):

        self.a = a

        self.b = b

    def __call__(self, x):

        """x: ndarray of shape (1, )"""

        if isinstance(x[0], float):

            return( np.tanh( self.a * x + self.b ) - x )

        elif isinstance(x[0], pyibex.Interval):

            return( vtanh(self.a * x + self.b) - x )

    def jacobian(self, x):

        """x: ndarray of shape (1, 1)"""

        if isinstance(x[0], float):

            return( self.a * 1/(np.cosh(self.a * x + self.b))**2 - 1. )

        elif isinstance(x[0], pyibex.Interval):

            return( self.a * 1/(vcosh(self.a * x + self.b))**2 - 1. )

a = np.arange(1.33, 10., 0.01)
b = np.arange(0.06, 5., 0.01)

x = np.linspace(-1., 1., 100)
z = np.zeros(100)

for i in range(100):

    init = np.array([Interval(-1., 1.)])

    queue = Queue()

    queue.append(init)

    f = Tanh(a[i], b[i])
    #f = Tanh(2.19, 0.6)

    verified = interval_bisection(f, queue)

    lbs = [(f.jacobian(interval) + 1)[0].lb() for interval in verified if interval[0].ub() < 0.]

    if len(lbs) == 2:

        print(lbs[0] * lbs[1])
    #fig, ax = plt.subplots()

    #ax.plot(x, f(x), x, z, x, f.jacobian(x) + 1)

    #plt.show()
