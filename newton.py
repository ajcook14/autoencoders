from pyibex import *
from tubex_lib import *

import numpy as np

from net import Net



mid = np.vectorize(lambda x: x.mid())

intersects = np.vectorize(lambda x, y: x.intersects(y))

isub = np.vectorize(lambda x, y: x.is_subset(y))

intersection = np.vectorize(lambda x, y: x & y)



def inverse(m): # m is a 2 x 2 matrix possibly of intervals

    a = m[0, 0]
    b = m[0, 1]
    c = m[1, 0]
    d = m[1, 1]

    det = a * d - b * c

    if det.contains(0):

        raise ZeroDivisionError

    inverse = np.array([[d, -b], [-c, a]]) / det

    return(inverse)


def newton(f, x):
    """
    Inputs:
    f: instance of DiffAE class
    x: numpy array of pyibex intervals (ie. a box)

    Output: a 2-tuple, out
    out[0]: an integer flag of result type
    out[1]: new refined interval if out[0] == 3
    """

    global mid, intersects, isub, intersection

    while True:

        try:

            inv = inverse(f.jacobian(x))

        except ZeroDivisionError:

            return((4, None))
        
        N_f = mid(x) - np.dot( inv, f(mid(x)) )

        if not (False in isub(N_f, x)): # N_f \subseteq x

            return((0, None))

        elif not (False in isub(x, N_f)): # x \subset N_f

            return((1, None))

        elif False in intersects(N_f, x):

            return((2, None))

        else:

            x = intersection(N_f, x)

            return((3, x))

"""
import numpy as np

from net import Net
from parameters import Parameters

import gzip
import pickle

from diffae import DiffAE


f = gzip.open('./data/quadratic/20210330_232952', 'rb')
params = pickle.load(f)
f.close()

parameters = (params.weights, params.biases)
net = Net(params.layers, parameters=parameters)


f = DiffAE(net)

x = np.array([Interval(0.45, 0.5), Interval(0.45, 0.5)])
#x = np.array([Interval(0, 1), Interval(0, 1)])

print(newton(f, x))

print(x)
"""
