import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import gzip, pickle, argparse

import activations

marker_size = mpl.rcParams['lines.markersize'] ** 2

parser = argparse.ArgumentParser()
parser.add_argument('file_name', metavar='YYYYMMDD_HHMMSS', type=str, nargs='+',
                    help='file name of the pickled parameters object')

args = parser.parse_args()

file_name = args.file_name[0]

aname = 'sigmoid'

f = gzip.open(f'./data/fixed_points/layers/{aname}/{file_name}', 'rb')

tup = pickle.load(f)

if len(tup) == 3:

    fixed_points, layers, seed = tup

    limits = (-20.0, 20.0, -20.0, 20.0)

    activation = activations.sigmoid

elif len(tup) == 4:

    fixed_points, layers, seed, limits = tup

    activation = activations.sigmoid

elif len(tup) == 5:

    fixed_points, layers, seed, limits, activation = tup

else:

    raise IndexError("tuple should have 3 or 4 elements")

f.close()

weights_lower, weights_upper, biases_lower, biases_upper = limits

print(f'layers = {layers}')


L = len(layers) - 1

rng = np.random.default_rng(seed)

data = []

for iteration in range(len(fixed_points)):

    parameters = []
    projected = []

    for i in range(L):

        temp_weights = rng.uniform(weights_lower, weights_upper, (layers[i + 1], layers[i]))
        temp_biases = rng.uniform(biases_lower, biases_upper, layers[i + 1])

        if i == 0:

            parameters.append( temp_weights[0,0] )
            parameters.append( temp_weights[1,0] )
            #parameters.append( temp_biases[0] )

        else:

            projected.append( temp_weights[0,0] )
            projected.append( temp_weights[0,1] )

    #if -10 < projected[1] < 0 and -10 < projected[0] < 0:

    data.append((np.array(parameters), fixed_points[iteration]))



one = []
two = []
three = []

for point in data:

    if point[1] == 1:

        one.append(point[0].reshape(2,1))

    elif point[1] == 2:

        two.append(point[0].reshape(2,1))

    elif point[1] == 3:
        
        three.append(point[0].reshape(2,1))

print(len(one))
print(len(two))
print(len(three))

one_data = np.hstack(one)
three_data = np.hstack(three)

fig, ax = plt.subplots()

ax.scatter(one_data[0,:], one_data[1,:], s=marker_size/4, marker='x', label='1 fixed point')
ax.scatter(three_data[0,:], three_data[1,:], s=marker_size/4, marker='x', label='3 fixed points')
ax.set_title('<<title>>')
ax.set_xlabel('<<x-axis>>')
ax.set_ylabel('<<y-axis>>')
ax.legend()

plt.show()
