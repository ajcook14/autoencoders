import numpy as np

class DiffAE():

    def __init__(self, net):

        assert net.layers[0] == net.layers[-1], "net is not an autoencoder."

        self.dim = net.layers[0]

        self.net = net

    def __call__(self, x):

        return(self.net.feedforward(x) - x)

    def jacobian(self, x):

        I = np.eye(self.dim)

        return(self.net.jacobian(x) - I)

