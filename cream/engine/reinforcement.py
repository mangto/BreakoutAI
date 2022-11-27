import numpy

import cream.tool.Csys as Csys
from cream.Functions import *
import numba

from numba.core.errors import NumbaWarning, NumbaDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)

class network:

    InputType = [list, numpy.array, numpy.ndarray]

    def __str__(self):
        result = f'''
        | type: Cream Neural Network
        | Network Type: reinforcement
        | Network Shape: {self.NetworkShape}
        | Activation Function: {self.acfunc}
        | Learning Rate: {self.lrate}
        '''

        return result

    def check(self):
        result = f'''
        weights: {self.weights}
        biases : {self.biases}
        '''
        return result

    def init_weights(NetworkShape:list):
        # initialize weights with network shape

        result = [[numpy.random.normal(size=NetworkShape[i])* 0.1 for j in range(shape)]
                    for i, shape in enumerate(NetworkShape[1:])] 

        return result

    def init_biases(NetworkShape:list):
        # initialize biases with network shape

        # result = [numpy.random.randn(shape)*0.1 for shape in NetworkShape[1:]]
        result = [numpy.zeros(shape) for shape in NetworkShape[1:]]

        return result

    def reset_activation(NetworkShape):
        # reset activations

        result = [numpy.zeros(shape) for shape in NetworkShape]

        return result
        
    def load_weight(self, weight):
        self.weights = [numpy.array(w) for w in weight]

    def load_bias(self, bias):
        self.biases = [numpy.array(b) for b in bias]

    def train_advice(self, text, epoch, error, lrate):
        result = text.replace("/epoch/", str(epoch)).replace("/error/", str(error)).replace("/lrate/", str(lrate))
        return result

    def __init__(self, NetworkShape:list, ActivationFunction=sigmoid, LearningRate:float=0.01,
                    weights:numpy.array=None, biases:numpy.array=None):

        self.NetworkShape = NetworkShape
        self.acfunc = ActivationFunction
        self.lrate = LearningRate

        self.weights = weights if weights else network.init_weights(NetworkShape)
        self.biases = biases if biases else network.init_biases(NetworkShape)

        self.activ = network.reset_activation(NetworkShape)
        self.raw_activ = network.reset_activation(NetworkShape)

        self.depth = len(NetworkShape)
        self.start_nb = self.check()

        self.fitness = 0

    def forward(self, input:list):
        assert type(input) in network.InputType, "Wrong Type of Input"
        assert len(input) == self.NetworkShape[0], f"Wrong Count of Input, need: {self.NetworkShape[0]} taken: {len(input)}"

        self.activ = network.reset_activation(self.NetworkShape)
        self.raw_activ = network.reset_activation(self.NetworkShape)
        self.activ[0] = input
        self.raw_activ[0] = input

        for i in range(len(self.NetworkShape[1:])):
            raw = numpy.sum(numpy.array(self.weights[i]) * numpy.array(self.activ[i]), axis=1) + self.biases[i]

            self.raw_activ[i+1] = raw
            self.activ[i+1] = self.acfunc(raw)