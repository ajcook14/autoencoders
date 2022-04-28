from net import Net
import activations
import numpy as np
from interval_bisection import interval_bisection
from pyibex import Interval
from queue import Queue
from diffae import DiffAE
import copy
import sys
import time
import gzip, pickle

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def euclidean(net1, net2):

    """
    returns the Euclidean distance between net1 and net2 in parameter space.
    net1 and net2 must have the same architecture.
    """

    distance = 0.

    assert net1.layers == net2.layers, "net1 and net2 must have the same architecture"

    params1 = net1.weights, net1.biases
    params2 = net2.weights, net2.biases

    L = len(net1.layers) - 1

    for l in range(L):

        distance += np.sum( (params1[0][l] - params2[0][l])**2 ) # weights
        distance += np.sum( (params1[1][l] - params2[1][l])**2 ) # biases

    return( np.sqrt(distance) )

def midpoint(net1, net2):

    """
    returns the midpoint of net1 and net2 in parameter space.
    net1 and net2 must have the same architecture.
    """

    net1 = copy.deepcopy(net1)
    net2 = copy.deepcopy(net2)

    weights = []
    biases = []

    assert net1.layers == net2.layers, "net1 and net2 must have the same architecture"

    params1 = net1.weights, net1.biases
    params2 = net2.weights, net2.biases

    L = len(net1.layers) - 1

    for l in range(L):

        weights.append( (params1[0][l] + params2[0][l])/2 )
        biases.append( (params1[1][l] + params2[1][l])/2 )

    layers = net1.layers

    parameters = (weights, biases)

    mid = Net(layers, parameters=parameters, activation=activations.sigmoid)

    return(mid)

def bisection(net1, net2, tol=1e-2):

    """
    Inputs
    net1: Net instance with 1 fixed point
    net2: Net instance with 3 fixed points
    tol: maximum length of the final 'interval'
    Output
    an instance of Net, close to a net with 2 fixed points
    """

    queue = Queue()

    init = np.array([Interval(0, 1)] * layers[0])

    while( euclidean(net1, net2) > tol ):

        mid = midpoint(net1, net2)

        # compute the number of fixed points
        queue.clean()

        queue.append(init)

        f = DiffAE(mid)

        verified = interval_bisection(f, queue)

        if len(verified) == 1:

            net1 = mid

        elif len(verified) == 3:

            net2 = mid

        elif len(verified) == 2:

            return(mid)

        else:

            print("midpoint has more than 3 fixed points")

    return(mid)

layers = [1, 2, 1]

p1 = (0, 0, (0, 0)) # these are the indices for the plotted parameters in the form
p2 = (0, 0, (1, 0)) # (weight (0)/bias (1), layer, indices as a 2-tuple)
p3 = (0, 1, (0, 1))

L = len(layers) - 1

max_fixed_points = 3

plot_data = [[] for _ in range(max_fixed_points)]   # the i-th inner list contains the parameters of
                                                    # all the neural networks that have i+1 fixed points

queue = Queue()

init = np.array([Interval(0, 1)] * layers[0])

for i in range(40):

    # randomly generate a neural network

    np_seed = i

    rng = np.random.default_rng(np_seed)

    weights = []
    biases = []

    for i in range(L):

        weights.append( np.ones((layers[i + 1], layers[i])) )
        biases.append( np.zeros((layers[i + 1], 1)) )

    parameters = (weights, biases)

    generated = [0., 0., 0.]
    for i in range(3):

        p = [p1, p2, p3][i]
        if p[0] == 0:

            generated[i] = rng.uniform(-40.,40.) # range for weights

        elif p[0] == 1:

            generated[i] = rng.uniform(-40.,40.) # range for biases

        else:

            print('strange behaviour in plotted parameter indexing')
    
    generated = tuple(generated)
    parameters[p1[0]][p1[1]][p1[2]], parameters[p2[0]][p2[1]][p2[2]], parameters[p3[0]][p3[1]][p3[2]] = generated

    net = Net(layers, parameters=parameters, activation=activations.sigmoid)



    # compute the number of fixed points
    queue.clean()

    queue.append(init)

    f = DiffAE(net)

    verified = interval_bisection(f, queue)

    if verified == -1:

        print(f'iteration {i} unverified')

    num_fixed_points = len(verified)

    if num_fixed_points > max_fixed_points:

        print(f'iteration {i} contains {num_fixed_points} fixed points')

    else:

        #point = np.array(list(weights[0].flatten()) + [weights[1].flatten()[1]])
        point = net
        plot_data[num_fixed_points - 1].append(point)

example_net = plot_data[num_fixed_points - 1][-1]
example = example_net.weights, example_net.biases

if len(plot_data[1]) > 0:

    #twos = np.stack(plot_data[1])
    print(f'hit bifurcation, {len(plot_data)} neural nets with 2 fixed points!')

n = min(len(plot_data[0]), len(plot_data[2]))

assert n > 0, 'no pair of generated networks with 1 and 3 fixed points'

manifold = []

start = time.time()
for i in range(n):

    print(f'\riteration {i} of {n}', end='')
    sys.stdout.flush()

    net = bisection(plot_data[0][i], plot_data[2][i])

    parameters = net.weights, net.biases

    point = np.array([parameters[p1[0]][p1[1]][p1[2]], parameters[p2[0]][p2[1]][p2[2]], parameters[p3[0]][p3[1]][p3[2]]])

    manifold.append(point)

print(f'\ncompleted in {time.time() - start} seconds')

manifold = np.stack(manifold)

fname = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
fname += '_' + '-'.join(map(lambda x: str(x), layers))

f = gzip.open(f'../data/manifold/{fname}', 'wb')

pickle.dump((layers, manifold, (p1, p2, p3), example), f)
# example is a parameters tuple which determines the values of the fixed parameters

f.close()

print(f'saved to file {fname}')
