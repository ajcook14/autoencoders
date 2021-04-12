import numpy as np
from net import Net
from parameters import Parameters
import gzip
import pickle
import os

folder = 'quadratic'
files = os.listdir(path=f'./data/{folder}/')

i = 0
for name in files:

    print(i)
    f = gzip.open(f'./data/{folder}/{name}', 'rb')
    net = pickle.load(f)
    f.close()

    params = Parameters(net.layers, -1, -1, -1, -1, -1, parameters = (net.weights, net.biases), training_info="migrated from old class")

    os.remove(f'./data/{folder}/{name}')

    f = gzip.open(f'./data/{folder}/{name}', 'wb')

    pickle.dump(params, f)

    f.close()

    i += 1
