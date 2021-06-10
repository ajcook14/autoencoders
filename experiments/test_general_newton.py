from pyibex import *
import numpy as np

from diffae import DiffAE
from interval_bisection import interval_bisection
from queue import Queue



sigmoid_interval = lambda z: 1/(1 + exp(-z))

sigmoid = np.vectorize(sigmoid_interval)

class Objective():

    def __init__(self):

        pass

    def __call__(self, x):

        return(sigmoid(x) - 0.5)

    def jacobian(self, x):

        sig = sigmoid(x)

        result = sig * (1 - sig)

        return(result.reshape((1,1)))



f = Objective()
R = np.array([Interval()])
queue = Queue()
queue.append(R)

verified = interval_bisection(f, queue)

print(verified)
