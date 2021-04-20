from pyibex import *
from tubex_lib import *

import numpy as np

from net import Net


def inverse(m): # m is a 2 x 2 matrix possibly of intervals

    a = m[0, 0]
    b = m[0, 1]
    c = m[1, 0]
    d = m[1, 1]

    det = a * d - b * c

    inverse = np.array([[d, -b], [-c, a]]) / det

    return(inverse)

def newton(f, x):

    mid = np.vectorize(lambda x: x.mid())

    intersects = np.vectorize(lambda x, y: x.intersects(y))

    isub = np.vectorize(lambda x, y: x.is_subset(y))

    intersection = np.vectorize(lambda x, y: x & y)

    while True:
        
        N_f = mid(x) - np.dot( inverse(f.jacobian(x)), f(mid(x)) )

        if not (False in isub(N_f, x)):

            return(x)

        elif not (False in isub(x, N_f)):

            return(-1)

        elif False in intersects(N_f, x):

            return(None)

        else:

            x = intersection(N_f, x)

"""
import numpy as np

from net import Net
from parameters import Parameters

import gzip
import pickle


f = gzip.open('./data/quadratic/20210330_232952', 'rb')
params = pickle.load(f)
f.close()

parameters = (params.weights, params.biases)
net = Net(params.layers, parameters=parameters)


f = DiffAE(net)

x = np.array([Interval(0.5, 0.51), Interval(0.5, 0.51)])
#x = np.array([Interval(0, 1), Interval(0, 1)])

newton(f, x)
"""
