from pyibex import *
from tubex_lib import *

from net import Net
import numpy as np
import copy

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from queue import Queue

from newton import newton


def delta(interval):

    return(interval.diam())

def min_dim(interval):

    if isinstance(interval, pyibex.pyibex.Interval):

        return( delta(interval) )

    elif isinstance(interval, np.ndarray):

        return( np.min(list(map(delta, interval))) )

    else:

        print("Interval is not an ndarray or tubex interval.")

        return(None)

def arg_max_dim(interval):

    if isinstance(interval, pyibex.pyibex.Interval):

        return( delta(interval) )

    elif isinstance(interval, np.ndarray):

        return( np.argmax(list(map(delta, interval))) )

    else:

        print("Interval is not an ndarray or tubex interval.")

        return(None)

def divide(interval):

    if isinstance(interval, pyibex.pyibex.Interval):

        return(interval.bisect(0.5))

    mindex = arg_max_dim(interval)

    mcomp = interval[mindex]

    split = mcomp.bisect(0.5)

    result = (copy.deepcopy(interval), copy.deepcopy(interval))

    result[0][mindex] = split[0]
    result[1][mindex] = split[1]

    return(result)

# set f(x) = net.feedforward(x) - x
def interval_bisection(f, queue):
    """
    Notes:
    Does not currently account for rounding errors in Python.
    Output:
    verified: list of small 'intervals' (boxes) covering all fixed points, 
        each box containing exactly one fixed point.
    """

    verified = []

    while not queue.is_empty():

        interval = queue.serve()

        image = f(interval)
        kill = False
        for comp in image:

            if not comp.contains(0.0):

                kill = True

                break

        if kill:

            continue

        #mindex = arg_max_dim(interval)
        #if interval[mindex].diam() < tol:

        #    verified.append(interval)

        result = newton(f, interval)

        if result[0] == 0: # N_f \subseteq x

            verified.append(interval)

        elif result[0] == 1 or result[0] == 4: # x \subset N_f or jacobian is singular

            split = divide(interval)

            for half in split:

                for element in list(half):

                    element.inflate(1e-17)

            queue.append(split[0])
            queue.append(split[1])

        elif result[0] == 2: # x \cap N_f == \phi

            pass

        elif result[0] == 3: # x \cap N_f \neq \phi

            queue.append(result[1])

    return(verified)

def rectangles(ax, intervals):

    tol = 0.01

    boxes = [Rectangle((i[0][0], i[1][0]), i[0].diam(), i[1].diam(), linewidth=2) if min_dim(i) > tol\
    else Rectangle((i[0].mid() - tol/2, i[1].mid() - tol/2), tol, tol, linewidth=2) for i in intervals]

    pc = PatchCollection(boxes, facecolor='c', alpha=0.5, edgecolor='r')

    ax.add_collection(pc)

