import numpy as np
from neuron import Neuron
from activation_functions import *
from typing import List
from cost_functions import CostFunction
from optimisation import OptimisationFunction

class Layer:
    def __init__(
                 self,
                 index: int,
                 dimension: tuple,
                 activation_function: ActivationFunction,
                 optimiser: OptimisationFunction
                ):
        assert index >= 0, "Layer index must be a non-negative integer"
        assert isinstance(index, int), "Layer index must be an integer"
        assert isinstance(dimension, tuple), "Layer dimensions must be a tuple"
        assert np.all([isinstance(dim, int) for dim in dimension]), "Layer dimensions must contain only integers"
        assert isinstance(activation_function, ActivationFunction), "Activation function must be an instance of ActivationFunction"
        assert isinstance(optimiser, OptimisationFunction), "Optimiser must be an instance of OptimisationFunction"
        input_size, neuron_count = dimension
        self.index: int = index
        self.activation_function: ActivationFunction = activation_function
        self.neurons: List[Neuron] = [Neuron(index, i, input_size, neuron_count) for i in range(neuron_count)]
        self.output = None
        self.input = None
        self.z_outputs = None
        self.delta = None
        self.children: List[Neuron] = []
        self.parents: List[Neuron] = []
        if optimiser == 'Adam':
            optimiser.map_neurons(self.neurons)
        
    def set_neuron_parents(self, parents: list):
        assert isinstance(parents, list), "Parents must be a list"
        assert np.all([isinstance(parent, Neuron) for parent in parents]), "Parents must be a list of Neurons"
        for neuron in self.neurons:
            neuron.set_parents(parents)

    def set_neuron_children(self, children: list):
        assert isinstance(children, list), "Children must be a list"
        assert np.all([isinstance(child, Neuron) for child in children]), "Children must be a list of Neurons"
        for neuron in self.neurons:
            neuron.set_children(children)

    def forward(self, input: np.ndarray):
        assert isinstance(input, np.ndarray), "Input must be a numpy array"
        assert len(input) > 0, "Input cannot be empty"
        self.input = input
        z_outputs = []
        for neuron in self.neurons:
            neuron_z_output = neuron.forward(input)
            z_outputs.append(neuron_z_output)
        self.z_outputs = np.column_stack(z_outputs)  # Shape: (batch_size, num_neurons)

        self.output = self.activation_function(self.z_outputs)

        # Update each neuron's output
        for neuron in self.neurons:
            neuron.output = self.output[:, neuron.index]

        return self.output

    def final_layer_backward(self, 
                             predictions: np.ndarray, 
                             target: np.ndarray, 
                             optimisation_function: OptimisationFunction, 
                             cost_function: CostFunction
                            ):
        assert isinstance(predictions, np.ndarray) and isinstance(target, np.ndarray), "Predictions and target must be numpy arrays"
        assert len(predictions) > 0 and len(target) > 0, "Predictions and target cannot be empty"
        assert predictions.shape == target.shape, "Predictions and target must have the same shape"
        assert isinstance(optimisation_function, OptimisationFunction), "Optimisation function must be an instance of OptimisationFunction"
        assert isinstance(cost_function, CostFunction), "Cost function must be an instance of CostFunction"
        
        batch_size: int = predictions.shape[0]
        previous_layer_output: np.ndarray = self.input

        # Compute delta for the output layer
        self.delta = cost_function.gradient(target, predictions) * self.activation_function.gradient(self.z_outputs)
        
        # Update each neuron's delta and compute gradients
        for neuron in self.neurons:
            neuron.delta = self.delta[:, neuron.index]
            gradient, bias_gradient = previous_layer_output.T.dot(neuron.delta) / batch_size, np.mean(neuron.delta)
            gradient = np.clip(gradient, -5, 5)  # Clip gradients to prevent exploding gradients
            optimisation_function(neuron, (gradient, bias_gradient))

    def backpropagate(self, optimisation_function: OptimisationFunction):
        assert isinstance(optimisation_function, OptimisationFunction), "Optimisation function must be an instance of OptimisationFunction"
        batch_size: int = self.output.shape[0]
        previous_layer_output = self.input

        # Compute delta for this layer
        self.delta = np.zeros_like(self.z_outputs)  # Shape: (batch_size, num_neurons)

        for neuron in self.neurons:
            self.delta[:, neuron.index] = neuron.compute_delta(self.activation_function)
            neuron.delta = self.delta[:, neuron.index]
            # Compute gradients and update weights and biases for each neuron
            gradient, bias_gradient = previous_layer_output.T.dot(neuron.delta) / batch_size, np.mean(neuron.delta)
            optimisation_function(neuron, (gradient, bias_gradient))
