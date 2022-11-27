import numpy
from cream.Functions.activation_functions import *

from pprint import pprint

class cbow:
    def __init__(self, input_set_count, window_size, projection_layer_count):
        self.input_set_count = input_set_count
        self.window_size = window_size
        self.projection_layer_count = projection_layer_count

        self.result = numpy.zeros((input_set_count,))

        self.weights = [
            numpy.random.randn(input_set_count, projection_layer_count),
            numpy.random.randn(input_set_count, projection_layer_count)
            ]

    def forward(self, input_sets:list):
        assert len(input_sets) == self.window_size*2, "wrong input sets' count"
        assert [len(i) for i in input_sets] == [self.input_set_count] * self.window_size*2, "invalid input"

        self.result = numpy.zeros((self.input_set_count,))

        result = [numpy.array(self.weights[0][numpy.argmax(array)]).tolist() for array in input_sets]
        middle = numpy.sum(result, axis=0) / (self.window_size * 2)
        result = numpy.sum(softmax(self.weights[1] * middle),axis=1)

        self.result = result
        return result

    def cross_entorpy(self, answer, result:numpy.array=None):
        return numpy.array(result := result if result else self.result - answer)