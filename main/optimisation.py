import numpy as np
from neuron import Neuron
from abc import ABC, abstractmethod

class OptimisationFunction(ABC):
    @abstractmethod
    def __call__(self, weights: np.ndarray, gradient: np.ndarray): ...

class GradientDescent(OptimisationFunction):
    def __init__(self, learning_rate=1.0):
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
    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0
        self.it = 0

    def __call__(self, weights: np.ndarray, gradient: np.ndarray):
        self.it += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient ** 2
        m_hat = self.m / (1 - self.beta1 ** self.it)
        v_hat = self.v / (1 - self.beta2 ** self.it)
        weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)