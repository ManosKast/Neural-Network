import numpy as np
from neuron import Neuron
from activation_functions import *
from typing import List

class Layer:
    def __init__(
                 self,
                 index: int,
                 dimension: tuple,
                 activation_function: str
                ):
        input_size, neuron_count = dimension
        self.index = index
        self.activation_function, self.activation_function_gradient = get_activation_function(activation_function)
        self.neurons: List[Neuron] = [Neuron(index, i, input_size, neuron_count) for i in range(neuron_count)]
        self.output = None
        self.input = None
        self.z_outputs = None
        self.delta = None
        self.children = []
        self.parents = []

    def set_neuron_parents(self, parents: list):
        for neuron in self.neurons:
            neuron.set_parents(parents)

    def set_neuron_children(self, children: list):
        for neuron in self.neurons:
            neuron.set_children(children)

    def forward(self, input):
        self.input = input
        z_outputs = []
        for neuron in self.neurons:
            neuron_z_output = neuron.forward(input)
            z_outputs.append(neuron_z_output)
        self.z_outputs = np.column_stack(z_outputs)  # Shape: (batch_size, num_neurons)

        self.output = self.activation_function(self.z_outputs)

        # Update each neuron's output
        for idx, neuron in enumerate(self.neurons):
            neuron.output = self.output[:, idx]

        return self.output

    def final_layer_backward(self, predictions: np.ndarray, target: np.ndarray, optimisation_function):
        batch_size: int = predictions.shape[0]
        previous_layer_output: np.ndarray = self.input

        # Compute delta for the output layer
        # TODO: Implement proper delta calculation for different loss functions
        self.delta = predictions - target

        # Update each neuron's delta and compute gradients
        for neuron in self.neurons:
            neuron.delta = self.delta[:, neuron.index]
            gradient = previous_layer_output.T.dot(neuron.delta) / batch_size
            gradient = np.clip(gradient, -5, 5)  # Clip gradients to prevent exploding gradients
            optimisation_function(neuron.weights, gradient)
            neuron.bias -= optimisation_function.learning_rate * np.mean(neuron.delta)

    def backpropagate(self, optimisation_function):
        batch_size = self.output.shape[0]
        previous_layer_output = self.input

        next_layer = self.children[0]

        # Compute delta for this layer
        self.delta = np.zeros_like(self.z_outputs)  # Shape: (batch_size, num_neurons)

        for idx, neuron in enumerate(self.neurons):
            weighted_deltas = np.zeros(batch_size)
            for child_idx, child_neuron in enumerate(next_layer.neurons):
                weighted_deltas += child_neuron.weights[idx] * child_neuron.delta
            self.delta[:, idx] = self.activation_function_gradient(self.z_outputs[:, idx]) * weighted_deltas
            neuron.delta = self.delta[:, idx]

        # Compute gradients and update weights and biases for each neuron
        for idx, neuron in enumerate(self.neurons):
            gradient = previous_layer_output.T.dot(neuron.delta) / batch_size
            optimisation_function(neuron.weights, gradient)
            neuron.bias -= optimisation_function.learning_rate * np.mean(neuron.delta)