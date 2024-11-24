import numpy as np

from layer import Layer
from cost_functions import *
from activation_functions import *
#from optimisation import *
from typing import List

class GradientDescent:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def __call__(self, weights, gradient):
        weights -= self.learning_rate * gradient
        self.learning_rate *= 0.999  # Decay learning rate
    
# TODO: Remove num_classes and add np.uniqiue(target) in the model function
# Converts an array of class labels into one-hot-encoded vectors.
def one_hot_encode(labels, num_classes):
    # Create a zero matrix of shape (num_samples, num_classes)
    one_hot = np.zeros((len(labels), num_classes))
    # Set the index corresponding to the label to 1 for each sample
    one_hot[np.arange(len(labels)), labels] = 1
    
    return one_hot



class NeuralNetwork:
    def __init__(self, cost_function='cross_entropy', optimisation_function=None):
        self.layers: List[Layer] = []
        self.cost_function = cross_entropy_loss
        self.optimisation_function = GradientDescent() if optimisation_function is None else optimisation_function
        self.predictions = None

    def add_layer(self, layer_dimensions: tuple, activation_function: str):
        layer = Layer(len(self.layers), layer_dimensions, activation_function)
        if len(self.layers) > 0:
            # Set parents and children at the neuron level
            layer.set_neuron_parents(self.layers[-1].neurons)
            self.layers[-1].set_neuron_children(layer.neurons)
            # Set parents and children at the layer level
            self.layers[-1].children.append(layer)
            layer.parents.append(self.layers[-1])
        self.layers.append(layer)

    def model(self, input: np.ndarray, target: np.ndarray):
        num_classes = len(np.unique(target))
        target_one_hot = one_hot_encode(target, num_classes)

        layer_input = input
        for layer in self.layers:
            layer_input = layer.forward(layer_input)
        self.predictions = layer_input

        loss = self.cost_function(target_one_hot, self.predictions)

        predictions_labels = np.argmax(self.predictions, axis=1)
        accuracy = np.mean(predictions_labels == target)
        return loss, accuracy

    def backpropagate(self, target: np.ndarray):
        num_classes = self.predictions.shape[1]
        target_one_hot = one_hot_encode(target, num_classes)
        self.layers[-1].final_layer_backward(self.predictions, target_one_hot, self.optimisation_function)
        for layer in reversed(self.layers[:-1]):
            layer.backpropagate(self.optimisation_function)

if __name__ == '__main__':
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])  # XOR input
    y = np.array([0, 1, 1, 0])  # XOR output

    network = NeuralNetwork()
    network.add_layer((2, 4), 'tanh')
    network.add_layer((4, 2), 'softmax')  # Output layer with 2 neurons

    epochs = 20000
    for epoch in range(epochs):
        loss, accuracy = network.model(X, y)
        network.backpropagate(y)
        if epoch % 1000 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
