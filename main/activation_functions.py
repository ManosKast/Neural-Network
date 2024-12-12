import numpy as np
from abc import ABC, abstractmethod

# Cost functions

def validate_positive_number(x):
    if x <= 0:
        raise ValueError('c must be greater than 0')
    if not isinstance(x, (int, float)):
        raise TypeError('c must be a number')

class ActivationFunction(ABC):
    def __init__(self):
        self.name = None

    @abstractmethod
    def __call__(self, x): 
        if not isinstance(x, (int, float, np.ndarray)):
            raise TypeError('x must be a number or numpy array')

    @abstractmethod
    def gradient(self, x): 
        if not isinstance(x, (int, float, np.ndarray)):
            raise TypeError('x must be a number or numpy array')

    def __eq__(self, value):
        if not isinstance(value, str):
            raise TypeError('value must be a string')
        return self.name == value


class Linear(ActivationFunction):
    def __init__(self):
        self.name = 'linear_activation'

    def __call__(self, x):
        super().__call__(x)
        return x

    def gradient(self, x=0.0):
        super().gradient(x)
        return 1


class BinaryStep(ActivationFunction):
    def __init__(self):
        self.name = 'binary_step_activation'

    def __call__(self, x):
        super().__call__(x)
        return 1 if x >= 0 else 0

    def gradient(self, x=0.0):
        super().gradient(x)
        return 0


class ReLU(ActivationFunction):
    def __init__(self):
        self.name = 'ReLU_activation'

    def __call__(self, x):
        super().__call__(x)
        return np.maximum(0, x)

    def gradient(self, x):
        super().gradient(x)
        return np.where(x > 0, 1.0, 0.0)


class PReLU(ActivationFunction):
    def __init__(self, alpha=0.01):
        validate_positive_number(alpha)
        self.name = 'PReLU_activation'
        self.alpha = alpha

    def __call__(self, x):
        super().__call__(x)
        return np.maximum(0, x) + self.alpha * np.minimum(0, x)

    def gradient(self, x):
        super().gradient(x)
        return 1 if x >= 0 else self.alpha


class ELU(ActivationFunction):
    def __init__(self, alpha=1.0):
        validate_positive_number(alpha)
        self.name = 'ELU_activation'
        self.alpha = alpha

    def __call__(self, x):
        super().__call__(x)
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))

    def gradient(self, x):
        super().gradient(x)
        return 1 if x >= 0 else self.alpha * np.exp(x)


class SLU(ActivationFunction):
    def __init__(self, c=1.0):
        validate_positive_number(c)
        self.name = 'SLU_activation'
        self.c = c

    def __call__(self, x):
        super().__call__(x)
        return x / (1 + np.exp(-self.c*x))

    def gradient(self, x):
        super().gradient(x)
        return np.exp(self.c*x) * (np.exp(self.c*x) + self.c*x + 1) / (np.exp(self.c*x) + 1)**2


class Softplus(ActivationFunction):
    def __init__(self):
        self.name = 'softplus_activation'

    def __call__(self, x):
        super().__call__(x)
        return np.log(1 + np.exp(x))

    def gradient(self, x):
        super().gradient(x)
        return 1 / (1 + np.exp(-x))


class Logistic(ActivationFunction):
    def __init__(self, c=1.0):
        validate_positive_number(c)
        self.name = 'logistic_activation'
        self.c = c

    def __call__(self, x):
        super().__call__(x)
        return 1 / (1 + np.exp(-self.c*x))

    def gradient(self, x):
        super().gradient(x)
        return self.c * np.exp(-self.c*x) / (1 + np.exp(-self.c*x))**2


class Tanh(ActivationFunction):
    def __init__(self):
        self.name = 'tanh_activation'

    def __call__(self, x):
        super().__call__(x)
        return np.tanh(x)

    def gradient(self, x):
        super().gradient(x)
        return 1 - np.tanh(x)**2


class Arctan(ActivationFunction):
    def __init__(self):
        self.name = 'arctan_activation'

    def __call__(self, x):
        super().__call__(x)
        return np.arctan(x)

    def gradient(self, x):
        super().gradient(x)
        return 1 / (x**2 + 1)


class Softmax(ActivationFunction):
    def __init__(self, cross_entropy=False):
        if not isinstance(cross_entropy, bool):
            raise TypeError('cross_entropy must be a boolean')
        self.cross_entropy = cross_entropy
        self.name = 'softmax_activation'

    def __call__(self, x):
        super().__call__(x)
        epsilon = 1e-10  # A small value to ensure numerical stability
        x_max = np.max(x, axis=1, keepdims=True)
        x_stable = np.clip(x - x_max, a_min=None, a_max=20)  # Clip to prevent overflow in exp
        exps = np.exp(x_stable) + epsilon
        return exps / np.sum(exps, axis=1, keepdims=True)

    def gradient(self, x):
        super().gradient(x)
        return 1 if self.cross_entropy else x * (1 - x)
