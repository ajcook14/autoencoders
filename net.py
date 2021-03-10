import numpy as np
import random
import copy
import matplotlib.pyplot as plt

class Net():

    def __init__(self, layers, parameters=None):

        """
        Here, layers includes the input and output layers.
        parameters (if given): a 2-tuple of lists of numpy.ndarrays representing pre-defined
            weights and biases for testing purposes.
        """

        L = len(layers) - 1
        assert L > 0, "must have at least one layer, excluding the input layer"

        self.layers = layers

        if parameters is None:

            self.weights = []
            self.biases = []

            for i in range(L):

                self.weights.append(np.random.rand(layers[i + 1], layers[i]))

                self.biases.append(np.random.rand(layers[i + 1]))

        else:

            self.weights = parameters[0]
            self.biases = parameters[1]


    def sigmoid(self, z):

        return( 1/(1 + np.e**(-z)) )

    def d_sigmoid(self, z):

        return( self.sigmoid(z) * (1 - self.sigmoid(z)) )

    def feedforward(self, x):

        for i in range(len(self.layers) - 1):

            x = self.sigmoid(np.dot(self.weights[i], x) + self.biases[i])

        return(x)

    def cost(self, x, label):

        return( np.sum((self.feedforward(x) - label)**2) )



    def SGD(self, training_data, epochs, size_minibatch, eta):
        """
        training_data: list of tuples of type (<class 'numpy.ndarray'>, <class 'numpy.ndarray'>), where
            the entries obey (input, output)
        size_minibatch: <class 'int'>
        """

        size_data = len(training_data)

        data_shuffled = copy.deepcopy(training_data)

        random.shuffle(data_shuffled)

        for epoch in range(epochs):

            #print("epoch %d"%(epoch + 1))

            for i in range(size_data // size_minibatch):

                minibatch = data_shuffled[i*size_minibatch:(i+1)*size_minibatch]

                self.backpropagation(minibatch, eta)

            # perform backpropagation for remainding training samples
            n = size_data // size_minibatch

            remainder_index = (size_data // size_minibatch) * size_minibatch

            if (remainder_index < size_data - 1):

                minibatch = data_shuffled[n*size_minibatch:]

                self.backpropagation(minibatch, eta)

            # validation to be added here

    def backpropagation(self, minibatch, eta):
        """
        minibatch: see training_data in self.SGD (a list of tuples)
        """

        # start with only updating the weights and biases in the last layer
        # start by assuming there is only one layer (excluding the input layer)

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



