import numpy as np
import matplotlib.pyplot as plt

import gzip, pickle

f = gzip.open('./data/fixed_points/20210512_180303', 'rb')

fixed_points, layers, seed = pickle.load(f)

f.close()
print(layers)



L = len(layers) - 1

rng = np.random.default_rng(seed)

data = []

for iteration in range(len(fixed_points)):

    weights = []
    projected = []

    for i in range(L):

        temp = rng.uniform(-10.0, 10.0, (layers[i + 1], layers[i]))

        if i == 0 or i == 1:

            weights.append( temp[0,0] )

        else:

            projected.append( temp[0,0] )

    if 0 < projected[0] < 1:

        data.append((np.array(weights), fixed_points[iteration]))


print(len(data))
print(data[-1])
"""
data_points = map(lambda x: [x[0][
fig, ax = plt.subplots()

ax.scatter(
"""
