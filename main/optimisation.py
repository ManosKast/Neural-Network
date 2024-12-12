import numpy as np
from neuron import Neuron
from abc import ABC, abstractmethod

class OptimisationFunction(ABC):
    @abstractmethod
    def __call__(self, weights: np.ndarray, gradient: np.ndarray): ...
    def __eq__(self, value):
        if not isinstance(value, str):
            raise TypeError("Expected value to be a string")
        return self.name == value

class GradientDescent(OptimisationFunction):
    def __init__(self, learning_rate=0.01):
        self.name = 'Gradient_Descent'
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.iteration = 0

    def __call__(self, neuron: Neuron, gradients: tuple[np.ndarray, np.ndarray]):
        if not isinstance(neuron, Neuron):
            raise TypeError("Expected neuron to be of type")
        if not isinstance(gradients, tuple):
            raise TypeError("Expected gradients to be a tuple")
        if not isinstance(gradients[0], np.ndarray) and not isinstance(gradients[1], (np.ndarray, int, float)):
            raise TypeError("Expected gradients to be a tuple of numpy arrays")
        gradient, bias_gradient = gradients
        neuron.weights -= self.learning_rate * gradient
        neuron.bias -= self.learning_rate * bias_gradient
        # TODO: Maybe add iteration in neuron class
        #self.learning_rate = self.initial_learning_rate * np.exp(-self.iteration * 0.00001)
        self.iteration += 1

class Adam(OptimisationFunction):
    def __init__(self, learning_rate=1e-2, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.name = 'Adam'
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.it = 1
        self.neurons = {}

    def __call__(self, neuron: Neuron, gradients: np.ndarray):
        #print(f'gradients: {gradients}')
        if not isinstance(neuron, Neuron):
            raise TypeError("Expected neuron to be of type Neuron")
        if not isinstance(gradients, (np.ndarray, tuple)):
            raise TypeError("Expected gradient to be a numpy array or tuple")
        gradient, bias_gradient = gradients
        #self.it += 1
        self.neurons[neuron]['m_1'] = self.beta1 * self.neurons[neuron]['m_1'] + (1 - self.beta1) * gradient
        self.neurons[neuron]['v_1'] = self.beta2 * self.neurons[neuron]['v_1'] + (1 - self.beta2) * np.power(gradient, 2)
        m_1_hat = self.neurons[neuron]['m_1'] / (1 - np.power(self.beta1, self.it))
        v_1_hat = self.neurons[neuron]['v_1'] / (1 - np.power(self.beta2, self.it))
        neuron.weights -= self.learning_rate * m_1_hat / (np.sqrt(v_1_hat) + self.epsilon)

        self.neurons[neuron]['m_2'] = self.beta1 * self.neurons[neuron]['m_2'] + (1 - self.beta1) * bias_gradient
        self.neurons[neuron]['v_2'] = self.beta2 * self.neurons[neuron]['v_2'] + (1 - self.beta2) * np.power(bias_gradient, 2)
        m_2_hat = self.neurons[neuron]['m_2'] / (1 - np.power(self.beta1, self.it))
        v_2_hat = self.neurons[neuron]['v_2'] / (1 - np.power(self.beta2, self.it))
        neuron.bias -= self.learning_rate * m_2_hat / (np.sqrt(v_2_hat) + self.epsilon)

    def map_neurons(self, neurons: list[Neuron]):
        for neuron in neurons:
            self.neurons[neuron] = {'m_1': 0, 'v_1': 0, 'm_2': 0, 'v_2': 0, 'it': 0}