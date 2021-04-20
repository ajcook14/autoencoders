from pyibex import *
from tubex_lib import *

import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import sys
import time
import os

class Net():

    def __init__(self, layers, seeds=None, parameters=None):

        """
        Here, layers includes the input and output layers.
        parameters (if given): a 2-tuple of lists of numpy.ndarrays representing pre-defined
            weights and biases for testing purposes.
        """

        L = len(layers) - 1
        assert L > 0, "must have at least one layer, excluding the input layer"

        self.layers = layers

        if parameters is None:

            if seeds is None:

                self.seed = os.urandom(8)
                self.np_seed = int(time.time() * 10**7)

            else:

                self.seed = seeds[0]
                self.np_seed = seeds[1]

            rng = np.random.default_rng(self.np_seed)

            self.weights = []
            self.biases = []

            for i in range(L):

                self.weights.append( rng.standard_normal((layers[i + 1], layers[i])) )

                self.biases.append( rng.standard_normal(layers[i + 1]) )

        else:

            assert seeds is None, "Seeds and parameters are provided"

            self.seed = None
            self.np_seed = None

            self.weights = parameters[0]
            self.biases = parameters[1]

        self.sigmoid_interval = lambda z: 1/(1 + exp(-z))

        self.vsigmoid_interval = np.vectorize(self.sigmoid_interval)

    def sigmoid(self, z):

        if isinstance(z, np.ndarray):

            if isinstance(z.flat[0], pyibex.pyibex.Interval):

                return( self.vsigmoid_interval(z) )

            else: # assume numpy can handle it

                return( 1/(1 + np.exp(-z)) )

        elif isinstance(z, pyibex.pyibex.Interval):

            return( self.sigmoid_interval(z) )

        else:

            return( 1/(1 + np.exp(-z)) )

    def d_sigmoid(self, z):

        return( self.sigmoid(z) * (1 - self.sigmoid(z)) )

    def feedforward(self, x):

        for i in range(len(self.layers) - 1):

            x = self.sigmoid(np.dot(self.weights[i], x) + self.biases[i])

        return(x)

    def cost(self, x, label):

        return( np.sum((self.feedforward(x) - label)**2) )



    def SGD(self, training_data, epochs, size_minibatch, size_validation, eta):
        """
        inputs:
        training_data: list of tuples of type (<class 'numpy.ndarray'>, <class 'numpy.ndarray'>), where
            the entries obey (input, output). Includes validation data.
        size_minibatch: <class 'int'>
        size_validation: <class 'int'>, the number of data points to be set aside for validation in
            each epoch. Taken from training_data once shuffled.

        output:
        validation_costs: <class 'numpy.ndarray'>
        """

        size_total = len(training_data)

        if not self.seed is None:

            random.seed(self.seed)

        data_shuffled = copy.deepcopy(training_data)

        random.shuffle(data_shuffled)

        assert size_validation < size_total, "Validation set is bigger than or equal to the whole dataset."
        validation_data = data_shuffled[:size_validation]
        data_shuffled = data_shuffled[size_validation:]

        validation_costs = []

        size_data = size_total - size_validation

        alpha = 0.1
        avg = 0
        last_duration = 0
        eta_p = eta
        try:
            for epoch in range(epochs):
                time.sleep(0.001)

                avg = (1 - alpha) * avg + alpha * last_duration
                overall_sec = int(avg * (epochs - epoch))
                sec  = overall_sec % 60
                mins = (overall_sec // 60) % 60
                hrs  = ((overall_sec // 60) // 60) % 24
                days = ((overall_sec // 60) // 60) // 24
                print("\rEpoch %7d of %7d, estimated %3d:%2d:%2d:%2d (ddd:hh:mm:ss) left"%(epoch + 1, epochs, days, hrs,\
                mins, sec), end='')

                sys.stdout.flush()

                start_time = time.time()
                for i in range(size_data // size_minibatch):

                    minibatch = data_shuffled[i*size_minibatch:(i+1)*size_minibatch]

                    self.backpropagation(minibatch, eta_p)

                # perform backpropagation for remainding training samples
                n = size_data // size_minibatch

                remainder_index = n * size_minibatch

                if (remainder_index < size_data - 1):

                    minibatch = data_shuffled[remainder_index:]

                    self.backpropagation(minibatch, eta_p)

                # validation
                cost_total = 0

                for sample_i in range(size_validation):

                    sample = validation_data[sample_i]

                    cost_total += self.cost(sample[0], sample[1])

                cost_avg = cost_total / size_validation

                validation_costs.append(cost_avg)

                print(", validation_loss = %3.3f"%(cost_avg,), end='')
                sys.stdout.flush()

                eta_p = eta#* cost_avg * 30

                # timing
                finish_time = time.time()
                last_duration = finish_time - start_time
        except KeyboardInterrupt:
            pass

        print('')

        return(np.array(validation_costs))

    def backpropagation(self, minibatch, eta):
        ########### NEEDS FIXING: indexing issue with case of 1 layer #############
        """
        minibatch: see training_data in self.SGD (a list of tuples)
        """

        m = len(minibatch)
        L = len(self.layers) - 1  # actual number of layers

        weights_avg = []
        biases_avg = []

        for layer in range(L):

            weights_avg.append(np.zeros_like(self.weights[layer]))
            biases_avg.append(np.zeros_like(self.biases[layer]))  # these are cumulative over the minibatch

        for i in range(m):  # iterate over training samples

            x = minibatch[i][0]

            y = minibatch[i][1]

            # forward propagation
            z = [np.dot(self.weights[0], x) + self.biases[0]]

            a = [self.sigmoid(z[0])]

            for j in range(L - 1):  # L - 1 because we have already evaluated the first layer

                z.append(np.dot(self.weights[j + 1], a[-1]) + self.biases[j + 1])

                a.append(self.sigmoid(z[-1]))

            # back propagation
            dC_da = [None] * L  # will be derivatives with respect
                                # to layer-wise activations for one
                                # specific training data point
            dC_da[L - 1] = 2 * (a[L - 1] - y)

            dC_dz = (dC_da[L - 1] * self.d_sigmoid(z[L - 1])).reshape((1, self.layers[L])).T

            weights_avg[L - 1] += np.dot( dC_dz, a[L - 2].reshape((1, self.layers[L - 1])) )
            biases_avg[L - 1] += dC_da[L - 1] * self.d_sigmoid(z[L - 1])

            for l in range(L - 2, 0, -1):  # L - 2: one for indexing from 0, one for taking above into account

                dC_da[l] = np.dot((dC_da[l + 1] * self.d_sigmoid(z[l + 1])).flatten(), self.weights[l + 1])
                
                dC_dz = (dC_da[l] * self.d_sigmoid(z[l])).reshape((1, self.layers[l + 1])).T

                weights_avg[l] += np.dot( dC_dz, a[l - 1].reshape((1, self.layers[l])) )
                biases_avg[l] += dC_da[l] * self.d_sigmoid(z[l])

            # because of a[] starting at layer 1, this is a special case
            dC_da[0] = np.dot((dC_da[1] * self.d_sigmoid(z[1])).flatten(), self.weights[1])
            
            dC_dz = (dC_da[0] * self.d_sigmoid(z[0])).reshape((1, self.layers[1])).T

            weights_avg[0] += np.dot( dC_dz, x.reshape((1, self.layers[0])) )
            biases_avg[0] += dC_da[0] * self.d_sigmoid(z[0])

        assert len(weights_avg) == len(biases_avg) == L, "weights_avg and biases_avg have different lengths"

        # update weights and biases
        for layer in range(L):

            weights_avg[layer] /= m
            biases_avg[layer] /= m

            self.weights[layer] = self.weights[layer] - eta * weights_avg[layer]
            self.biases[layer] = self.biases[layer] - eta * biases_avg[layer]

    def jacobian(self, x):

        L = len(self.layers) - 1  # actual number of layers

        # forward propagation

        a = x   # note this approach is different to that used in the training algorithm, since here
                # we can avoid a special case for the last layer in the backpropagation (first layer)
        z = []

        for j in range(L):

            z.append(np.dot(self.weights[j], a) + self.biases[j])

            a = self.sigmoid(z[-1])

        # back propagation

        dy_da = [None] * L

        dy_dz = self.d_sigmoid(z[L - 1])
        dy_dz = dy_dz.reshape((self.layers[L], 1))

        dy_da[L - 1] = dy_dz * self.weights[L - 1]

        for l in range(L - 2, -1, -1):

            da_dz = self.d_sigmoid(z[l])
            da_dz = da_dz.reshape((self.layers[l + 1], 1))

            da_da = da_dz * self.weights[l]

            dy_da[l] = np.dot( dy_da[l + 1], da_da )

        return(dy_da[0])

