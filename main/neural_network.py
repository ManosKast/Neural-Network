import numpy as np

from layer import Layer
from cost_functions import *
from activation_functions import *
from optimisation import *
from util import *

from typing import List

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler





class NeuralNetwork:
    def __init__(self, cost_function='cross_entropy', optimisation_function=None):
        assert isinstance(cost_function, (str, CostFunction)), "Cost function must be a string or an instance of CostFunction"
        if optimisation_function is not None and not isinstance(optimisation_function, (str, OptimisationFunction)):
            assert False, "Optimisation function must be a string or an instance of OptimisationFunction"
        self.cost_function = get_cost_function(cost_function) if isinstance(cost_function, str) else cost_function

        # If optimisation_function is None, use Gradient Descent as the default optimisation function
        # If optimisation_function is a string, get the corresponding optimisation function using get_optimisation_function
        # If optimisation_function is an instance of OptimisationFunction, assign it directly
        self.optimisation_function = GradientDescent() if optimisation_function is None else get_optimisation_function(optimisation_function) \
                                                       if isinstance(optimisation_function, str) else optimisation_function
        self.layers: List[Layer] = []
        self.predictions = None
        self.output_function = None

    def add_layer(self, layer_dimensions: tuple, activation_function: str):
        assert isinstance(layer_dimensions, tuple), "Layer dimensions must be a tuple"
        assert np.all([isinstance(dim, int) for dim in layer_dimensions]), "Layer dimensions must contain only integers"
        assert isinstance(activation_function, str), "Activation function must be a string"

        activation_function: ActivationFunction = get_activation_function(activation_function) 
        # TODO: Implement the following logic for logistic
        # If the parameter activation_function is softmax, set it as the output function   
        # If there are no layers, raise an exception, as the output layer cannot be the first layer
        if activation_function == 'softmax_activation':
            self.output_function = activation_function
            if len(self.layers) == 0:
                assert False, "Output layer cannot be the first layer"
        
        # Create a layer and if there are existing layers, then its parent
        # is the last layer in the list of layers. And vice versa for children.
        layer: Layer = Layer(len(self.layers), layer_dimensions, activation_function, self.optimisation_function)
        if len(self.layers) > 0:
            layer.set_neuron_parents(self.layers[-1].neurons)
            self.layers[-1].set_neuron_children(layer.neurons)
            self.layers[-1].children.append(layer)
            layer.parents.append(self.layers[-1])
        self.layers.append(layer)

    def model(self, input: np.ndarray, target: np.ndarray):
        assert input is not None and target is not None, "Input and target cannot be None"
        assert len(input) > 0 and len(target) > 0, "Input and target cannot be empty"
        assert isinstance(input, np.ndarray) and isinstance(target, np.ndarray), "Input and target must be numpy arrays"
        
        # If an output function has not been explicitly set, add a softmax output layer
        if self.output_function is None:
            classes_count, input_count = len(np.unique(target)), len(self.layers[-1].neurons)
            self.add_layer((input_count, classes_count), 'softmax')
        
        # If the output function is softmax and the cost function is cross-entropy,
        # set the softmax and cross-entropy flags to True, so that the final layer's delta
        # can be computed easily by deducting y_pred from y.
        if self.output_function == 'softmax_activation' and self.cost_function == 'cross_entropy':
            self.cost_function.softmax, self.output_function.cross_entropy = True, True

        target_one_hot: np.ndarray = one_hot_encode(target)

        # Forward pass. Compute each layer's output and forward it to the next layer.
        # The last layer will contain the predictions always.
        layer_input = input
        for layer in self.layers:
            layer_input = layer.forward(layer_input)
        self.predictions = layer_input

        loss = self.cost_function(target_one_hot, self.predictions)

        predictions_labels = np.argmax(self.predictions, axis=1)
        accuracy = np.mean(predictions_labels == target)
        return loss, accuracy

    def backpropagate(self, target: np.ndarray):
        assert target is not None, "Target cannot be None"
        assert len(target) > 0, "Target cannot be empty"
        assert isinstance(target, np.ndarray), "Target must be a numpy array"
        target_one_hot = one_hot_encode(target)
        self.layers[-1].final_layer_backward(self.predictions, target_one_hot, self.optimisation_function, self.cost_function)
        for layer in reversed(self.layers[:-1]):
            layer.backpropagate(self.optimisation_function)

    def predict(self, input: np.ndarray):
        assert input is not None, "Input cannot be None"
        assert len(input) > 0, "Input cannot be empty"
        assert isinstance(input, np.ndarray), "Input must be a numpy array"
        layer_input = input
        for layer in self.layers:
            layer_input = layer.predict(layer_input)
        return np.argmax(layer_input, axis=1)


if __name__ == '__main__':
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the dataset
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert the data to NumPy arrays
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int64)

    network = NeuralNetwork(optimisation_function='adam')
    network.add_layer((4, 10), 'ReLU')
    network.add_layer((10, 3), 'softmax')

    epochs = 100
    batch_size = 16
    num_batches = len(X_train) // batch_size

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        for batch in range(num_batches):
            X_batch = X_train[batch * batch_size:(batch + 1) * batch_size]
            y_batch = y_train[batch * batch_size:(batch + 1) * batch_size]

            loss, accuracy = network.model(X_batch, y_batch)
            network.backpropagate(y_batch)

            epoch_loss += loss
            epoch_accuracy += accuracy

        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches

        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

    # Evaluate on test data
    predictions = network.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
    