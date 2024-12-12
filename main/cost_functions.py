import numpy as np
from abc import ABC, abstractmethod

c = 1e-10

class CostFunction(ABC):
    @abstractmethod
    def __call__(self, y, y_pred): ...
    @abstractmethod
    def gradient(self, y, y_pred): ...
    @abstractmethod
    def __eq__(self, value): ...

class L1Error(CostFunction):
    def __init__(self):
        self.name = 'l1_error'

    def __call__(self, y, y_pred):
        return np.sum(np.abs(y - y_pred))

    def gradient(self, y, y_pred):
        return np.sum((y_pred - y) / np.abs(y - y_pred + c))

    def __eq__(self, value):
        return self.name == value
    
class L2Error(CostFunction):
    def __init__(self):
        self.name = 'l2_error'

    def __call__(self, y, y_pred):
        return np.sum(np.power(y - y_pred, 2))

    def gradient(self, y, y_pred):
        return np.sum(2 * (y_pred - y))

    def __eq__(self, value):
        return self.name == value
    
class CrossEntropyLoss(CostFunction):
    def __init__(self, softmax=False):
        self.name = 'cross_entropy'
        self.softmax = softmax

    def __call__(self, y, y_pred):
        return -np.mean(y * np.log(y_pred + c))

    def gradient(self, y, y_pred):
        return y_pred - y if self.softmax else np.mean(y / (y_pred + c))

    def __eq__(self, value):
        return self.name == value
    
class MeanSquaredError(CostFunction):
    def __init__(self):
        self.name = 'mean_squared_error'

    def __call__(self, y, y_pred):
        return np.mean(np.power(y - y_pred, 2))

    def gradient(self, y, y_pred):
        return np.mean(2 * (y_pred - y))

    def __eq__(self, value):
        return self.name == value
    
class MeanAbsoluteError(CostFunction):
    def __init__(self):
        self.name = 'mean_absolute_error'

    def __call__(self, y, y_pred):
        return np.mean(np.abs(y - y_pred))

    def gradient(self, y, y_pred):
        return np.mean((y_pred - y) / np.abs(y - y_pred + c))

    def __eq__(self, value):
        return self.name == value
    
# TODO: Check if it classifies 2 classes
class BinaryCrossEntropy(CostFunction):
    def __init__(self):
        self.name = 'binary_cross_entropy'

    def __call__(self, y, y_pred):
        return np.mean(-y * np.log(y_pred + c) - (1 - y) * np.log(1 - y_pred + c))

    def gradient(self, y, y_pred):
        y_pred = np.clip(y_pred, c, 1 - c)
        gradient = -(y / y_pred) + (1 - y) / (1 - y_pred + c)
        return np.mean(gradient)

    def __eq__(self, value):
        return self.name == value
    
class HingeLoss(CostFunction):
    def __init__(self):
        self.name = 'hinge_loss'

    def __call__(self, y, y_pred):
        pass

    def gradient(self, y, y_pred):
        pass

    def __eq__(self, value):
        return self.name == value