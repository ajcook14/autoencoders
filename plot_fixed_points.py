import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import gzip, pickle, argparse

marker_size = mpl.rcParams['lines.markersize'] ** 2

parser = argparse.ArgumentParser()
parser.add_argument('file_name', metavar='YYYYMMDD_HHMMSS', type=str, nargs='+',
                    help='file name of the pickled parameters object')

args = parser.parse_args()

file_name = args.file_name[0]

f = gzip.open(f'./data/fixed_points/{file_name}', 'rb')

tup = pickle.load(f)

if len(tup) == 3:

    fixed_points, layers, seed = tup

    limits = (-20.0, 20.0, -10.0, 10.0)

elif len(tup) == 4:

    fixed_points, layers, seed, limits = tup

else:

    raise IndexError("tuple should have 3 or 4 elements")

f.close()

weights_lower, weights_upper, biases_lower, biases_upper = limits



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
            parameters.append( temp_biases[0] )

        else:

            projected.append( temp_weights[0,0] )
            projected.append( temp_weights[0] )

    #if -10 < projected[1] < 0 and -10 < projected[0] < 0:

    data.append((np.array(parameters), fixed_points[iteration]))


print(len(data))
print(data[-1])

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

ax.scatter(one_data[0,:], one_data[1,:], s=marker_size/4, marker='x')
ax.scatter(three_data[0,:], three_data[1,:], s=marker_size/4, marker='x')

plt.show()
