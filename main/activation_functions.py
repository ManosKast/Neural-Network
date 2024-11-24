import numpy as np
from abc import ABC, abstractmethod
#TODO: Add leaky RELU

# Cost functions

class ActivationFunction(ABC):
    @abstractmethod
    def __call__(self, x): ...
    @abstractmethod
    def gradient(self, x): ...
    @abstractmethod
    def __eq__(self, value): ...


class Linear(ActivationFunction):
    def __init__(self):
        self.name = 'linear_activation'

    def __call__(self, x):
        return x

    def gradient(self, x):
        return 1
    
    def __eq__(self, value):
        return self.name == value
    
class BinaryStep(ActivationFunction):
    def __init__(self):
        self.name = 'binary_step_activation'

    def __call__(self, x):
        return 1 if x >= 0 else 0

    def gradient(self, x):
        return 0
    
    def __eq__(self, value):
        return self.name == value
    
class ReLU(ActivationFunction):
    def __init__(self):
        self.name = 'ReLU_activation'

    def __call__(self, x):
        return np.maximum(0, x)

    def gradient(self, x):
        return np.where(x > 0, 1.0, 0.0)

class PReLU(ActivationFunction):
    def __init__(self, alpha=0.01):
        self.name = 'PReLU_activation'
        self.alpha = alpha

    # If x > 0, return x, else return alpha * x
    # If x > 0, np.minimum(0, x) = 0, so the function returns x
    # If x < 0, np.maximum(0, x) = 0, so the function returns alpha * x
    def __call__(self, x):
        return np.maximum(0, x) + self.alpha * np.minimum(0, x)

    def gradient(self, x):
        return 1 if x >= 0 else self.alpha
    
    def __eq__(self, value):
        return self.name == value
    
class ELU(ActivationFunction):
    def __init__(self, alpha=1.0):
        self.name = 'ELU_activation'
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))

    def gradient(self, x):
        return 1 if x >= 0 else self.alpha * np.exp(x)
    
    def __eq__(self, value):
        return self.name == value
    
class SLU(ActivationFunction):
    def __init__(self, c=1.0):
        self.name = 'SLU_activation'
        self.c = c

    def __call__(self, x):
        return x / (1 + np.exp(-self.c*x))

    def gradient(self, x):
        return np.exp(self.c*x) * (np.exp(self.c*x) + self.c*x + 1) / (np.exp(self.c*x) + 1)**2
    
    def __eq__(self, value):
        return self.name == value
    
class Softplus(ActivationFunction):
    def __init__(self):
        self.name = 'softplus_activation'

    def __call__(self, x):
        return np.log(1 + np.exp(x))

    def gradient(self, x):
        return 1 / (1 + np.exp(-x))
    
    def __eq__(self, value):
        return self.name == value
    
class Logistic(ActivationFunction):
    def __init__(self, c=1.0):
        self.name = 'logistic_activation'
        self.c = c

    def __call__(self, x):
        return 1 / (1 + np.exp(-self.c*x))

    def gradient(self, x):
        return self.c * np.exp(-self.c*x) / (1 + np.exp(-self.c*x))**2
    
    def __eq__(self, value):
        return self.name == value
    
class Tanh(ActivationFunction):
    def __init__(self):
        self.name = 'tanh_activation'

    def __call__(self, x):
        return np.tanh(x)

    def gradient(self, x):
        return 1 - np.tanh(x)**2
    
    def __eq__(self, value):
        return self.name == value
    
class Arctan(ActivationFunction):
    def __init__(self):
        self.name = 'arctan_activation'

    def __call__(self, x):
        return np.arctan(x)

    def gradient(self, x):
        return 1 / (x**2 + 1)
    
    def __eq__(self, value):
        return self.name == value
    
class Softmax(ActivationFunction):
    def __init__(self, cross_entropy=False):
        self.cross_entropy = cross_entropy
        self.name = 'softmax_activation'

    def __call__(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def gradient(self, x):
        return 1 if self.cross_entropy else x * (1 - x)
    
    def __eq__(self, value):
        return self.name == value


def get_activation_function(activation_function: str, hyperparameter: float = 0):
    match activation_function:
        case 'linear':
            return Linear()
        case 'binary_step':
            return BinaryStep()
        case 'ReLU':
            return ReLU()
        case 'PReLU':
            return PReLU() if hyperparameter == 0 else PReLU(hyperparameter)
        case 'ELU':
            return ELU() if hyperparameter == 0 else ELU(hyperparameter)
        case 'SLU':
            return SLU() if hyperparameter == 0 else SLU(hyperparameter)
        case 'softplus':
            return Softplus()
        case 'logistic':
            return Logistic() if hyperparameter == 0 else Logistic(hyperparameter)
        case 'tanh':
            return Tanh()
        case 'arctan':
            return Arctan()
        case 'softmax':
            return Softmax()
        case _:
            raise ValueError(f'Activation function {activation_function} not recognised')
