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
    def __init__(self, layer_count, index, input_size, output_size):
        # if activation_function == 'relu_activation':
        # self.weights = xavier_weight_initialisation(input_size, input_size, output_size)
        self.weights = kaiming_weight_initialisation(input_size)
        self.bias = 0.0
        self.output = None
        self.z_output = None
        self.delta = None
        self.children = []
        self.parents = []
        self.layer = layer_count
        self.index = index

    # Compute z = w * x + b
    def compute_z(self, input):
        return input.dot(self.weights) + self.bias
    
    def forward(self, input: np.ndarray):
        self.z_output = self.compute_z(input)
        return self.z_output  # Return z_output without activation

    def compute_final_delta(self, predictions, target):
        self.delta = predictions[:, self.index] - target[:, self.index]

    def compute_delta(self, activation_function):
        if not isinstance(activation_function, Arctan):
            assert("Activation function must be an instance of ActivationFunction")
        weighted_deltas = np.zeros_like(self.z_output)
        for child in self.children:
            weighted_deltas += child.weights[self.index] * child.delta
        self.delta = activation_function.gradient(self.z_output) * weighted_deltas

    def set_parents(self, parents: list):
        if isinstance(parents, list) and all(isinstance(parent, Neuron) for parent in parents):
            self.parents = parents
        else:
            assert("Parents must be a list of Neurons")

    def set_children(self, children: list):
        if isinstance(children, list) and np.all(isinstance(child, Neuron) for child in children):
            self.children = children
        else:
            assert("Children must be a list of Neurons")

    def set_output(self, output):
        if (isinstance(output, np.ndarray) and np.all(isinstance(o, (int, float)) for o in output)) or isinstance(output, (int, float)):
            self.output = output
        else:
            assert("Output must be a numpy array of floats or integers")