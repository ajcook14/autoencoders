

class Parameters():
    """
    Instance of a trained neural network, complete parameters required to
        replicate the training of the neural network.
    """

    def __init__(self, layers, training_data, epochs, size_minibatch, size_validation, eta, seeds=None, parameters=None, training_info=None):

        """
        layers, parameters: same as in Net().
        training_data, epochs, size_minibatch, size_validation, eta: same as in Net.SGD()
        seeds: tuple of type (<class 'bytes'>, <class 'int'>)
        NOTE: parameters and seeds are complimentary - exactly one of the two must be defined.
        training_info: optional argument - any additional details/descriptions of the training data
        """

        if seeds is None:

            assert not parameters is None, "Neither seeds nor parameters are provided"

        else:

            self.seed = seeds[0]
            self.np_seed = seeds[1]

        self.layers = layers

        self.weights = parameters[0]
        self.biases= parameters[1]

        self.training_data = training_data

        self.epochs = epochs

        self.size_minibatch = size_minibatch

        self.size_validation = size_validation

        self.eta = eta

        self.training_info = training_info
