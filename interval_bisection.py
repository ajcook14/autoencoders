from pyibex import *
from tubex_lib import *

from net import Net
import numpy as np
import copy

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from queue import Queue



def delta(interval):

    return(interval.diam())

def max_dim(interval):

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

    mindex = max_dim(interval)

    mcomp = interval[mindex]

    split = mcomp.bisect(0.5)

    result = (copy.deepcopy(interval), copy.deepcopy(interval))

    result[0][mindex] = split[0]
    result[1][mindex] = split[1]

    return(result)

# set f(x) = net.feedforward(x) - x
def interval_bisection(f, queue, tol):
    """
    Does not currently account for rounding errors in Python.
    Yet to prove the output intervals actually contain fixed points.
    Output:
    final: list of small 'intervals' (boxes) whose unions complement
        contains no fixed points.
    """

    final = []

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

        mindex = max_dim(interval)
        if interval[mindex].diam() < tol:

            final.append(interval)

        else:

            split = divide(interval)

            queue.append(split[0])
            queue.append(split[1])

    return(final)

def rectangles(ax, intervals):

    boxes = [Rectangle((i[0][0], i[1][0]), i[0].diam(), i[1].diam(), linewidth=2) for i in intervals]

    pc = PatchCollection(boxes, facecolor='c', alpha=0.5, edgecolor='r')

    ax.add_collection(pc)

