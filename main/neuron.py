import numpy as np
from typing import List
from activation_functions import *


def kaiming_weight_initialisation(count):
    weights = np.random.normal(0, np.sqrt(2 / count), count)
    #weights = np.random.normal(10, 1000, count)
    return weights

def xavier_weight_initialisation(size, fan_in, fan_out):
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(size) * std


class Neuron:
    def __init__(self, layer_count, index, input_size, output_size, activation_function):
        assert isinstance(layer_count, int), "Layer count must be an integer"
        assert isinstance(index, int), "Index must be an integer"
        assert isinstance(input_size, int), "Input size must be an integer"
        assert isinstance(output_size, int), "Output size must be an integer"
        assert isinstance(activation_function, ActivationFunction), "Activation function must be an instance of ActivationFunction"
        self.weights = kaiming_weight_initialisation(input_size) if activation_function == 'relu_activation' \
                                                                 else xavier_weight_initialisation(input_size, input_size, output_size)
        self.bias = 0.0
        self._output = None
        self.z_output = None
        self._delta = None
        self.children: List[Neuron] = []
        self.parents: List[Neuron] = []
        self.layer = layer_count
        self.index = index

    @property
    def delta(self):
        return self._delta
    
    @delta.setter
    def delta(self, value):
        assert isinstance(value, (np.ndarray, int, float)), "Delta must be a numpy array or a number"
        self._delta = value

    @property
    def output(self):
        return self._output
    
    @output.setter
    def output(self, value):
        assert isinstance(value, (np.ndarray, int, float)), "Output must be a numpy array or a number"
        self._output = value

    # Compute z = w * x + b
    def compute_z(self, input):
        return input.dot(self.weights) + self.bias
    
    def forward(self, input: np.ndarray):
        self.z_output = self.compute_z(input)
        return self.z_output  # Return z_output without activation
    
    def compute_final_delta(self, predictions, target):
        self._delta = predictions[:, self.index] - target[:, self.index]

    def compute_delta(self, activation_function: ActivationFunction):
        assert isinstance(activation_function, ActivationFunction), "Invalid activation function type"
        weighted_deltas = np.sum((child.weights[self.index] * child.delta for child in self.children), keepdims=True)
        self.delta = activation_function.gradient(self.z_output) * weighted_deltas
        return self.delta

    def set_parents(self, parents: list):
        if isinstance(parents, list) and all(isinstance(parent, Neuron) for parent in parents):
            self.parents = parents
        else:
            assert False, "Parents must be a list of Neurons"

    def set_children(self, children: list):
        if isinstance(children, list) and np.all(isinstance(child, Neuron) for child in children):
            self.children = children
        else:
            assert False, "Children must be a list of Neurons"

    def set_output(self, output):
        if (isinstance(output, np.ndarray) and np.all(isinstance(o, (int, float)) for o in output)) or isinstance(output, (int, float)):
            self.output = output
        else:
            assert False, "Output must be a numpy array or a number"